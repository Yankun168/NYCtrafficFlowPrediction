import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="cnnlstm",choices=["cnngru","gru","lstm","cnnlstm"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--drop_prob", type=float, default=0.5)
parser.add_argument("--iscuda",type=bool,default=False)
parser.add_argument("--hidden_size",type=int,default=64)

args = parser.parse_args()

if __name__ == "__main__":
    pass