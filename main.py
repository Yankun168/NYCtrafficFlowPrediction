import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import math
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.gru import MyGRU
from model.cnn_gru import CNNGRU
from model.lstm import MyLSTM
from model.cnn_lstm import CNNLSTM
from data_loader import NYCSTDNDataset
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from configuration import args
from func import nextBatch,mape,drawPlot

# ignore warning
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model,train_dataset,loss_fn,optimizer):
    model.train()
    iteration = 0
    for batch in nextBatch(shuffle(train_dataset.data),args.batch_size):
        x,y = batch[:,:-1,:], batch[:,-1,:]
        if args.iscuda:
            x,y = x.to(device),y.to(device)
        y_hat = model(x)
        l = loss_fn(y_hat, y)
        optimizer.zero_grad(set_to_none=True)
        l.backward()
        # # 梯度裁剪
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()
        iteration += 1
        if iteration % 10 == 0:
            print("Iteraion {}, Train Loss: {:.8f}".format(iteration, l.item()))
    
    train_rmse,train_mae,train_mape,train_loss = test(model,train_dataset,loss_fn)
    
    return (train_rmse,train_mae,train_mape,train_loss)

def test(model,test_dataset,loss_fn):
    model.eval()
    y_hats = []
    test_l_sum,c = 0,0
    with torch.no_grad():
        for batch in nextBatch(test_dataset.data, batch_size=args.batch_size):
            x,y = batch[:,:-1,:], batch[:,-1,:]
            if args.iscuda:
                x,y = x.to(device),y.to(device)
            y_hat = model(x)
            test_l_sum += loss_fn(y_hat,y).item()
            c += 1
            y_hats.append(y_hat.detach().cpu().numpy())
        y_hats = np.concatenate(y_hats)
    y_true = test_dataset.data[:,-1,:]
    y_hats = test_dataset.denormalize(y_hats)
    y_true = test_dataset.denormalize(y_true)
    y_true = y_true.reshape(y_true.size(0),-1)
    rmse_score,mae_score,mape_score = math.sqrt(mse(y_true, y_hats)), \
                                mae(y_true, y_hats), \
                                mape(y_true, y_hats)    
    return (rmse_score,mae_score,mape_score,test_l_sum / c)

if __name__ == "__main__":
    train_dataset,test_dataset = NYCSTDNDataset(data_path='NYC-stdn/volume_train.npz'),\
                                NYCSTDNDataset(data_path='NYC-stdn/volume_test.npz')
    print("loading dataset done")

    if args.model == "gru":
        model = MyGRU(input_size=400,
                      hidden_size=args.hidden_size,
                      output_size=400,
                      drop_prob=args.drop_prob)
    elif args.model == "lstm":
        model = MyLSTM(input_size=400,
                       hidden_size=args.hidden_size,
                       output_size=400,
                       drop_prob=args.drop_prob)
    elif args.model == "cnngru":
        model = CNNGRU(
                       in_channel=2,
                       out_channels=[2,2],
                       input_size=400,
                       hidden_size=args.hidden_size,
                       output_size=400,
                       drop_prob=args.drop_prob)
    elif args.model == "cnnlstm":
        model = CNNLSTM(
                       in_channel=2,
                       out_channels=[64,128],
                       input_size=400,
                       hidden_size=args.hidden_size,
                       output_size=400,
                       drop_prob=args.drop_prob)
    print("loading model {} done".format(args.model))


    optimizer =  optim.Adam(params=model.parameters(),lr=args.lr)
    loss_fn = nn.MSELoss()

    if torch.cuda.is_available():
        args.iscuda = True
        model = model.to(device)

    train_loss_list,test_loss_list = [],[]
    train_rmse_list,train_mae_list,train_mape_list = [],[],[]
    test_rmse_list,test_mae_list,test_mape_list = [],[],[]
    train_times = 0.0
    for i in range(args.epochs):
        print("=========epoch {}=========".format(i + 1))
        train_rmse,train_mae,train_mape,train_loss = train(model,train_dataset,loss_fn,optimizer)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_mape_list.append(train_mape)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f},Train Loss: {:.8f}'.format(
            i + 1,train_rmse,train_mae,train_mape,train_loss))
        # eval
        test_rmse,test_mae,test_mape,test_loss = test(model,test_dataset,loss_fn)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        test_mape_list.append(test_mape)
        print('Epoch: {}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f},Test Loss: {:.8f}'.format(
            i + 1,test_rmse,test_mae,test_mape,test_loss,train_loss,test_loss))

    with open("log.txt","a") as fp:
        fp.write("model:{}, lr: {}, epochs:{}, batch size:{}, hidden_size:{}, train times:{}\n".format(
            args.model,args.lr,args.epochs,args.batch_size,args.hidden_size,train_times))

    metrics = [train_loss_list,test_loss_list,train_rmse_list,test_rmse_list,
        train_mae_list,test_mae_list,train_mape_list,test_mape_list]
    # metrics curve
    fname = "{}_lr{}_b{}_h{}_d{}_metrics.png".format(args.model,args.lr,
        args.batch_size,args.hidden_size,args.drop_prob)
    drawPlot(metrics,fname,["loss","rmse","mae","mape"]) 

