import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from utils import split_data
from model import CNN_LSTM
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available!")


# 读取数据
def load_data(patient_id):
    file_path = f"/kaggle/input/subdata/sub{patient_id}.csv"
    return pd.read_csv(file_path)


# 数据预处理
def preprocess_data(data, time_step):
    # 将数据拆分为序列
    X, y = split_data(data.T, time_step)
    X = np.transpose(X, (0, 2, 1))  # 调整维度顺序
    return X, y


# 数据标准化
def standardize_data(X, y, scaler_data, scaler_label):
    # 重塑为二维数组
    X_2d = X.reshape(-1, X.shape[-1])
    y_2d = y.reshape(-1, 1)

    # 应用标准化
    X_scaled = scaler_data.fit_transform(X_2d)
    y_scaled = scaler_label.fit_transform(y_2d)

    # 恢复原始形状
    X_scaled = X_scaled.reshape(X.shape)
    return X_scaled, y_scaled


# 训练模型
def train_model(model, train_loader, test_loader, epochs, optimizer, criterion):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    outputs = model(inputs)
                    train_loss += criterion(outputs, labels).item()
                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                test_loss = 0
                for batch in test_loader:
                    inputs, labels = batch
                    outputs = model(inputs)
                    test_loss += criterion(outputs, labels).item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)

                print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses


# 主程序
def main():
    # 参数
    input_size = 10
    conv_input = 40
    time_step = 40
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 1500
    batch_size = 64
    learning_rate = 0.01
    betas = (0.5, 0.999)
    gamma = 0.3
    step_size = 100

    # 创建标准化器
    scaler_data = RobustScaler()
    scaler_label = RobustScaler()

    # 加载数据
    patient_data = {}
    for i in range(1, 19):
        key = f"patient_{i}"
        data = load_data(i)
        patient_data[key] = preprocess_data(data.iloc[:, :11].values, time_step)

    # 18折交叉验证
    mse_scores = []
    for patient_index in range(6, 19):
        print(f"Validation patient Index: {patient_index}")

        # 划分训练集和验证集
        train_data, train_labels = [], []
        for key, (X, y) in patient_data.items():
            if key != f"patient_{patient_index}":
                train_data.append(X)
                train_labels.append(y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)

        val_data, val_labels = patient_data[f"patient_{patient_index}"]

        # 标准化数据
        train_data, train_labels = standardize_data(train_data, train_labels, scaler_data, scaler_label)
        val_data, val_labels = standardize_data(val_data, val_labels, scaler_data, scaler_label)

        # 转换为张量
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32),
                                    torch.tensor(val_labels, dtype=torch.float32))

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size).to(device)

        # 定义优化器和学习率调度器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # 损失函数
        criterion = nn.MSELoss().to(device)

        # 训练模型
        train_losses, test_losses = train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion)

        # 绘制损失曲线
        plt.figure()
        plt.plot(range(0, num_epochs, 100), train_losses, marker="o", markersize=1, label="Train Loss")
        plt.plot(range(0, num_epochs, 100), test_losses, marker="x", markersize=1, label="Test Loss")
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 预测
        model.eval()
        with torch.no_grad():
            predictions = model(val_dataset.tensors[0].to(device)).detach().cpu().numpy()
            true_labels = val_dataset.tensors[1].numpy()

        # 计算评价指标
        mae = mean_absolute_error(predictions, true_labels)
        mse = mean_squared_error(predictions, true_labels)
        rmse = np.sqrt(mse)
        r2 = r2_score(predictions, true_labels)
        evs = explained_variance_score(predictions, true_labels)

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Explained Variance Score: {evs:.4f}")


if __name__ == '__main__':
    main()
