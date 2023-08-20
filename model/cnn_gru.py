import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRU(nn.Module):
    def __init__(self,in_channel,out_channels,input_size,hidden_size,output_size,drop_prob):
        super(CNNGRU,self).__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=2)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(768 + hidden_size,output_size)

    def forward(self,x):
        """
        x: (batch,depth/seqlen,channels*h*w)
        """
        x = x.view(x.shape[0],x.shape[1],2,10,20)
        cnn_feats = self.convs(x.transpose(1,2))
        print(cnn_feats.shape)
        gru_out,_ = self.gru(x.view(x.shape[0],x.shape[1],-1))
        gru_feats =torch.mean(gru_out,dim=1)
        fusion_feats = torch.cat((cnn_feats.view(cnn_feats.shape[0],-1),gru_feats),dim=1)
        x = self.fc(self.dropout(fusion_feats))
        return F.sigmoid(x)

if __name__ == "__main__":
    pass