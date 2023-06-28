# sound_event_detection
Sound event detection using CRNN and Res-conformer

--to run crnn model--

python run.py train_evaluate configs/baseline.yaml data/eval/feature.csv data/eval/label.csv 

--to run resconformer model--

python run.py train_evaluate configs/Res_conformer.yaml data/eval/feature.csv data/eval/label.csv 

Of course, you need data, which i have not upload it. 
