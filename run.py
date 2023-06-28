#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime

import uuid
import glob
from pathlib import Path
import fire

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skmetrics
from tabulate import tabulate

import dataset
import models
import utils
import metrics
import losses
import res_conformer

DEVICE = 'cpu'
if torch.cuda.is_available(
        ):
    DEVICE = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)


class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        aid, feat, target = batch
        feat = feat.to(DEVICE).float()
        target = target.to(DEVICE).float()
        output = model(feat)
        # print(output)
        output["aid"] = aid
        output["target"] = target
        return output

    def train(self, config_file, **kwargs):
        config = utils.parse_config_or_kwargs(config_file, **kwargs)
        outputdir = os.path.join(
            config['outputpath'], config['model']['type'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        # Create base dir
        Path(outputdir).mkdir(exist_ok=True, parents=True)

        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config, logger.info)
        logger.info("Running on device {}".format(DEVICE))

        label_to_idx = {}
        with open(config['data']['class_indice_file'], "r") as reader:
            for line in reader.readlines():
                idx, label = line.strip().split(",")
                label_to_idx[label] = int(idx)
        labels_df = pd.read_csv(config['data']['label'],
                                sep='\s+').convert_dtypes()
        label_array = labels_df["event_labels"].apply(lambda x: utils.encode_label(
            x, label_to_idx))
        label_array = np.stack(label_array.values)
        train_df, cv_df = utils.split_train_cv(
            labels_df, y=label_array, stratified=config["data"]["stratified"])

        utils.dump_config(os.path.join(outputdir, 'config.yaml'), config)

        trainloader = torch.utils.data.DataLoader(
            dataset.TrainDataset(
                config["data"]["feature"],
                train_df,
                label_to_idx
            ),
            collate_fn=dataset.sequential_collate(False),
            shuffle=True,
            **config["dataloader_args"]
        )

        cvdataloader = torch.utils.data.DataLoader(
            dataset.TrainDataset(
                config["data"]["feature"],
                cv_df,
                label_to_idx
            ),
            collate_fn=dataset.sequential_collate(False),
            shuffle=False,
            **config["dataloader_args"]
        )
        model = getattr(models, config['model']["type"])(
            num_freq=trainloader.dataset.datadim,
            num_class=len(label_to_idx),
            **config['model']['args'])
        # model = res_conformer.ResNetConformer(input_dim=trainloader.dataset.datadim,num_classes=len(label_to_idx))
        model = model.to(DEVICE)

        optimizer = getattr(torch.optim, config['optimizer']['type'])(
            model.parameters(), **config['optimizer']['args'])
        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        loss_fn = getattr(losses, config['loss'])().to(DEVICE)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config['scheduler_args'])
        
        not_improve_cnt = 0
        best_loss = float("inf")

        # Training
        for epoch in range(1, config['epochs'] + 1):

            model.train()
            loss_history = []
            with torch.enable_grad(), tqdm(total=len(trainloader), unit="batch", leave=False) as pbar:
                for batch in trainloader:
                    optimizer.zero_grad()
                    output = self._forward(model, batch)
                    loss = loss_fn(output)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
                    pbar.update()
            train_loss = np.mean(loss_history)

            model.eval()
            preds = []
            targets = []
            loss_history = []
            with torch.no_grad(), tqdm(total=len(trainloader), unit="batch", leave=False) as pbar:
                for batch in cvdataloader:
                    output = self._forward(model, batch)
                    loss = loss_fn(output)
                    loss_history.append(loss.item())
                    y_pred = output["clip_prob"]
                    y_pred = torch.round(y_pred)
                    preds.append(y_pred.cpu().numpy())
                    targets.append(output["target"].cpu().numpy())
                    pbar.update()
            val_loss = np.mean(loss_history)
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            p, r, f1, _ = skmetrics.precision_recall_fscore_support(
                targets, preds, average="macro",zero_division=0)
            logging_msg = f"Epoch {epoch}   training_loss: {train_loss:.2f}  val_loss: {val_loss:.2f}  " \
                          f"precision: {p:.2f}  recall: {r:.2f}  f1: {f1:.2f}"
            logger.info(logging_msg)

            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                not_improve_cnt = 0
                torch.save(model.state_dict(), os.path.join(outputdir, f"run_model_best.pth"))
            else:
                not_improve_cnt += 1

            if not_improve_cnt == config['early_stop']:
                break

        return outputdir

    def evaluate(
            self,
            experiment_path: str,
            feature: str,
            label: str,
            pred_file='predictions.csv',
            tag_file='tagging.txt',
            event_file='event.txt',
            segment_file='segment.txt',
            time_ratio=10. / 500,
            threshold=0.5,
            window_size=1):
        experiment_path = Path(experiment_path)
        config = utils.parse_config_or_kwargs(experiment_path / "config.yaml")
        state_dict = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],
            map_location="cpu")
        label_df = pd.read_csv(label, sep='\t')

        label_to_idx = {}
        idx_to_label = {}
        with open(config['data']['class_indice_file'], "r") as reader:
            for line in reader.readlines():
                idx, label = line.strip().split(",")
                label_to_idx[label] = int(idx)
                idx_to_label[int(idx)] = label
        
        dataloader = torch.utils.data.DataLoader(
            dataset.InferenceDataset(feature),
            batch_size=1,
            shuffle=False,
            num_workers=config["dataloader_args"]["num_workers"]
        )
        model = getattr(models, config['model']['type'])(
            num_freq=dataloader.dataset.datadim,
            num_class=len(label_to_idx),
            **config['model']['args'])
        # model = res_conformer.ResNetConformer(input_dim=dataloader.dataset.datadim,num_classes=len(label_to_idx))
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()

        clip_targets = []
        clip_probs = []
        frame_preds, clip_preds = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False):
                aids, feats = batch
                feats = feats.to(DEVICE).float()
                output = model(feats)
                frame_prob_batch = output["frame_prob"].cpu().numpy()
                clip_prob_batch = output["clip_prob"].cpu().numpy()
                filtered_pred = utils.median_filter(
                    frame_prob_batch, window_size=window_size, threshold=threshold)
                frame_pred_batch = utils.decode_with_timestamps(
                    idx_to_label, filtered_pred)

                for sample_idx in range(len(frame_pred_batch)):
                    aid = aids[sample_idx]

                    # clip results for mAP
                    clip_probs.append(clip_prob_batch[sample_idx])
                    clip_target = label_df.loc[label_df["filename"] == aid][
                        "event_label"].unique()
                    clip_targets.append(utils.encode_label(clip_target,
                        label_to_idx))
                    
                    # clip results after postprocessing
                    clip_pred = clip_prob_batch[sample_idx].reshape(1, -1)
                    clip_pred = utils.binarize(clip_pred)[0]
                    clip_pred = [idx_to_label[i] for i, tgt in 
                        enumerate(clip_pred) if tgt == 1]
                    for clip_label in clip_pred:
                        clip_preds.append({
                            'filename': aid,
                            'event_label': clip_label,
                            'probability': clip_prob_batch[sample_idx][
                                label_to_idx[clip_label]]
                        })

                    # time results after postprocessing
                    frame_pred = frame_pred_batch[sample_idx]
                    for event_label, onset, offset in frame_pred:
                        frame_preds.append({
                            'filename': aid,
                            'event_label': event_label,
                            'onset': onset,
                            'offset': offset
                        })

        assert len(frame_preds) > 0, "No outputs, lower threshold?"
        frame_pred_df = pd.DataFrame(frame_preds, columns=['filename', 'event_label',
            'onset', 'offset'])
        clip_pred_df = pd.DataFrame(clip_preds, columns=['filename',
            'event_label', 'probability'])
        frame_pred_df = utils.predictions_to_time(frame_pred_df, ratio=time_ratio)
        if pred_file:
            frame_pred_df.to_csv(os.path.join(experiment_path, pred_file),
                index=False, sep="\t", float_format="%.3f")
        tagging_df = metrics.audio_tagging_results(label_df, clip_pred_df,
            label_to_idx)

        clip_targets = np.stack(clip_targets)
        clip_probs = np.stack(clip_probs)
        average_precision = skmetrics.average_precision_score(np.array(clip_targets),
            np.array(clip_probs), average=None)
        print("mAP: {}".format(average_precision))
            
        if tag_file:
            tagging_df.to_csv(os.path.join(experiment_path, tag_file),
                index=False, sep='\t', float_format="%.3f")

        event_result, segment_result = metrics.compute_metrics(
            label_df, frame_pred_df, time_resolution=1.0)
        if event_file:
            with open(os.path.join(experiment_path, event_file), 'w') as wp:
                wp.write(event_result.__str__())
        if segment_file:
            with open(os.path.join(experiment_path, segment_file), 'w') as wp:
                wp.write(segment_result.__str__())
        event_based_results = pd.DataFrame(
            event_result.results_class_wise_average_metrics()['f_measure'],
            index=['event_based'])
        segment_based_results = pd.DataFrame(
            segment_result.results_class_wise_average_metrics()['f_measure'],
            index=['segment_based'])
        result_quick_report = pd.concat((
            event_based_results,
            segment_based_results,
        ))

        tagging_res = tagging_df.loc[
            tagging_df['label'] == 'macro'].values[0][1:]
        result_quick_report.loc['tagging_based'] = list(tagging_res)

        with open(os.path.join(experiment_path, 'quick_report.md'), 'w') as wp:
            print(tabulate(result_quick_report,
                           headers='keys',
                           tablefmt='github'), file=wp)
            print("mAP: {}".format(np.mean(average_precision)), file=wp)

        print("Quick Report: \n{}".format(
            tabulate(result_quick_report,
                     headers='keys',
                     tablefmt='github')))

    def train_evaluate(
        self, config_file, eval_feature, eval_label, **eval_kwargs):
        experiment_path = self.train(config_file)
        self.evaluate(experiment_path, eval_feature, eval_label, **eval_kwargs)


if __name__ == "__main__":
    fire.Fire(Runner)

