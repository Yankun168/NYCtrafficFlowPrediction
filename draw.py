import torch
import torch.nn as nn
import torchviz
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, in_channel, out_channels, input_size, hidden_size, output_size, drop_prob):
        super(CNNLSTM, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=2)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(384 + hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch,depth/seqlen,channels*h*w)
        """
        x = x.view(x.shape[0], x.shape[1], 2, 10, 20)
        cnn_feats = self.convs(x.transpose(1, 2))
        # print(cnn_feats.shape)
        gru_out, _ = self.lstm(x.view(x.shape[0], x.shape[1], -1))
        gru_feats = torch.mean(gru_out, dim=1)
        fusion_feats = torch.cat((cnn_feats.view(cnn_feats.shape[0], -1), gru_feats), dim=1)
        x = self.fc(self.dropout(fusion_feats))
        return F.sigmoid(x)

# 模型参数
in_channel = 2
out_channels = [64, 128]
input_size = 400  # 根据您的数据维度进行修改
hidden_size = 64
output_size = 400
drop_prob = 0.5
batch_size = 64
depth = 10
channels = 3
height = 20
width = 20

# 创建模型实例
model = CNNLSTM(in_channel, out_channels, input_size, hidden_size, output_size, drop_prob)

# 创建一个示例输入（需要根据你的模型输入维度来定义）
example_input = torch.randn(batch_size, depth, channels, height, width)

# 通过 make_dot 绘制模型图
torchviz.make_dot(model(example_input), params=dict(model.named_parameters())).view()
