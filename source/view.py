import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import fisher_exact
def test():
    # # 加载 .npy 文件
    # file_path = "/home/minheng/UTA/BrainNetworkTransformer/source/dataset/abide.npy"  # 替换为你的文件路径
    file_path="./adni.npy"
    data = np.load(file_path, allow_pickle=True).item()  # 设置 allow_pickle=True 以防保存的是复杂结构

    label = data["label"]
    corr = data["corr"]
    timeseries = data["timeseries"]
        
    # 打印数据和形状
    if label is not None:
        print("Label:", label)
        print("Label Shape:", np.shape(label))
    else:
        print("Label not found")
        
    if corr is not None:
        print("Corr:", corr)
        print("Corr Shape:", np.shape(corr))
    else:
        print("Corr not found")
        
    if timeseries is not None:
        print("Timeseries:", timeseries)
        print("Timeseries Shape:", np.shape(timeseries))
    else:
        print("Timeseries not found")

test()
