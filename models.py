import torch
import torch.nn as nn
import torch.nn.functional as F



def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        super(Crnn, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class

        self.bn1 = nn.BatchNorm1d(self.num_freq)
        self.convblk1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2))
        )
        self.convblk2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2))
        )
        self.convblk3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        self.convblk4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        self.convblk5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p=0.25),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
    
        self.bigru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, self.num_class)
        self.to('cuda')

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        x.to('cuda')
        x = x.permute(0, 2, 1) # (batch_size, num_freq, time_steps)
        x = self.bn1(x)
        x = x.permute(0, 2, 1) # (batch_size, time_steps, num_freq)
        x = x.unsqueeze(1) # (batch_size, 1, time_steps, num_freq)
        x = self.convblk1(x)     
        x = self.convblk2(x)    
        x = self.convblk3(x)      
        x = self.convblk4(x)     
        x = self.convblk5(x)
        
        #frequency mean
        x = x.mean(3)    
        x = F.interpolate(x, scale_factor = 4, mode='linear')
        x = x.permute(0, 2, 1)   
        #RNN
        x, _ = self.bigru(x)    
        # x = F.interpolate(x.permute(0, 2, 1), scale_factor = 4, mode='linear')
        # x = x.permute(0, 2, 1)
        x = torch.sigmoid(self.fc1(x))
        return x

    def forward(self, x):
        out = self.detection(x)
        frame_prob = out # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }


class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += 0.5 * identity
        out = self.relu(out)
        return out

##start Res_conformer##

class ConformerBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, num_heads=4, dropout=0.1):
        super(ConformerBlock, self).__init__()
        dim_feedforward = 2 * hidden_dim
        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, x):
        #feedward1
        residual = x
        x = self.feed_forward1(x)
        x = self.norm1(residual*0.5 + self.dropout(x))
        #multi-head attention
        residual = x
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2) 
        x = self.norm2(residual*0.5 + self.dropout(x))
        #convolution
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(residual*0.5 + self.dropout(x))
        #feedward2
        residual = x
        x = self.feed_forward2(x)
        x = self.norm4(residual*0.5 + self.dropout(x))
        return x


class ResNetConformer(nn.Module):
    def __init__(self, num_freq, num_class, num_resblks=5, num_confblks=5,hidden_dim=256, kernel_size=3, num_heads=4, dropout=0.1):
        super(ResNetConformer, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class

        self.bn1 = nn.BatchNorm1d(self.num_freq)
        # ResNet部分
        self.resnet = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 卷积层
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)), # 最大池化层
            self._make_layer(hidden_dim, num_resblks), # ResNet块
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Conformer部分
        self.conformer = nn.Sequential(
            *[ConformerBlock(hidden_dim, kernel_size, num_heads, dropout) for _ in range(num_confblks)]
        )
        

        # self.pooling = nn.MaxPool2d(kernel_size=(2, 1))
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_class)
        # self.fc1 = nn.Linear(hidden_dim, 128)
        # self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        # 输入维度：(batch_size, T, D)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)    
        x = x.permute(0, 2, 1)     
        x = x.unsqueeze(1) # 转换为4D张量：(batch_size, 1, T, D)
        x = self.resnet(x) # ResNet部分
        # print(x.shape) 
        x = x.mean(3)  
        # x = F.interpolate(x, scale_factor = 4, mode='linear')
        x = x.permute(0, 2, 1)   # 转换为3D张量：(batch_size, T, hidden_dim)
        # print(x.shape) 
        x = self.conformer(x) # Conformer部分,out(batch_size, T, hidden_dim)
        # print(x.shape) 
        # x = self.fc1(x)
        x = torch.sigmoid(self.fc(x)) # 全连接层

        # print(x.shape) 
        frame_prob = torch.nn.functional.interpolate(x.permute(0, 2, 1), scale_factor=(4),mode='linear').permute(0, 2, 1)
        # print(frame_prob.shape) 
        clip_prob = linear_softmax_pooling(frame_prob)

        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }

    def _make_layer(self, hidden_dim, num_resblks):
        layers = []
        for i in range(num_resblks): 
            layers.append(ResNetBlock(hidden_dim))
        return nn.Sequential(*layers)
