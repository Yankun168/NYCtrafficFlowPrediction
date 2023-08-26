import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,drop_prob):
        super(MyLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=2)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        """
        x: (batch,seq,feature)
        """
        x,_ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(torch.mean(x,dim=1))
        x = F.sigmoid(x)
        return x

if __name__ == "__main__":
    pass