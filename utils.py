import numpy as np

# 提取特征和标签
def split_data(data_raw, lookback):
    data = []
    # 自行设置样本包含的时间步
    for index in range(data_raw.shape[1] - lookback):
        data.append(data_raw[:, index: index + lookback])
    data = np.array(data)  # 将list转为numpy
    x_1 = data[:, 0:10, :]  # 前10列是特征
    y_1 = data[:, 10:11, -1]  # 11列是标签
    return [x_1, y_1]
