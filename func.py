import math
import numpy as np
import matplotlib.pyplot as plt

def drawPlot(heights,fname,ylabel):
    """
    功能：绘制训练集上的准确率和测试集上的loss和acc变化曲线
    heights: 纵轴值列表
    fname：保存的文件名
    """
    plt.figure(figsize=(9, 6))
    x = [i for i in range(1,len(heights[0]) + 1)]
    # 绘制训练集和测试集上的loss变化曲线子图
    plt.xlabel("epoch")
    # 设置横坐标的刻度间隔
    plt.xticks([i for i in range(0,len(heights[0]) + 1,5)])
    
    axe1 = plt.subplot(2,2,1)
    plt.ylabel(ylabel[0])
    axe1.plot(x,heights[0],label="train")
    axe1.plot(x,heights[1],label="test")
    axe1.legend()

    axe2 = plt.subplot(2,2,2)
    plt.ylabel(ylabel[1])
    axe2.plot(x,heights[2],label="train")
    axe2.plot(x,heights[3],label="test")
    plt.legend()

    axe3 = plt.subplot(2,2,3)
    plt.ylabel(ylabel[2])
    axe3.plot(x,heights[4],label="train")
    axe3.plot(x,heights[5],label="test")
    plt.legend()

    axe4 = plt.subplot(2,2,4)
    plt.ylabel(ylabel[3])
    axe4.plot(x,heights[6],label="train")
    axe4.plot(x,heights[7],label="test")
    plt.legend()

    plt.savefig("images/{}".format(fname))
    plt.show()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100

def nextBatch(data,batch_size):
    """
    Divide data into mini-batch
    """
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for idx in range(num_batches):
        start_idx = batch_size * idx
        end_idx = min(start_idx + batch_size, data_length)
        yield data[start_idx:end_idx]

if __name__ == "__main__":
    pass