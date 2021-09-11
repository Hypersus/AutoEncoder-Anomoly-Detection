from torch.functional import Tensor
import read as rd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.utils.data as dt
from sklearn.neighbors import KernelDensity
from scipy.stats import
class AutoEncoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(kwargs["input_size"], 40),
            nn.ReLU(),
            nn.Linear(40, 35),
            nn.ReLU(),
            nn.Linear(35, 31)
        )
        self.decoder=nn.Sequential(
            nn.Linear(31,35),
            nn.ReLU(),
            nn.Linear(35,40),
            nn.ReLU(),
            nn.Linear(40, kwargs["input_size"]),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded
# --------------------训练集位置--------------------
file_train = "F:\Data Science\实验室\data\d00.csv"
file_te ="F:\Data Science\实验室\data\d00_te.csv"
# --------------------读入.csv转换为numpy数组--------------------
raw_data_train= rd.read_csv(file=file_train)
raw_data_te = rd.read_csv(file=file_te)
# print(raw_data)
# --------------------归一化--------------------
scaler=StandardScaler()
data_train_scaler=scaler.fit_transform(raw_data_train)
data_te_scaler=scaler.transform(raw_data_te)
# print(data_scaler)
# --------------------numpy数组转化为tensor--------------------
data_train = torch.from_numpy(data_train_scaler)
data_train= data_train.to(torch.float32)
data_te=torch.from_numpy(data_te_scaler)
# print(data_train)
# print(data_train.shape)
# ----------------------超参数指定--------------------
Epochs = 100
Lr_Rate = 1e-3
Batch_Size = 100
# ----------------------训练集特征数--------------------
input_size=data_train.shape[1]
# print(input_size)
train_loader=dt.DataLoader(dataset=data_train,batch_size=Batch_Size,shuffle=True)
# ----------------------损失函数---------------------------
loss_fun=nn.MSELoss()
# 用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(input_size=input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Lr_Rate)
# -----------------------开始训练-------------------------
for epoch in range(Epochs):
    loss = 0
    for batch_features in train_loader:
        # load it to the active device
        batch_features = batch_features.to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        encode,decode = model(batch_features)
        
        # compute training reconstruction loss
        train_loss = loss_fun(decode, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, Epochs, loss))
print(encode)
# Tensor从gpu到cpu
temp=encode.cpu()
# cpu中的tensor转换为numpy数组
encode=temp.detach().numpy()
print(encode)
n=encode.shape[1]
# 协方差矩阵
# print(n)
# print(encode.shape[0])
S=np.matmul((np.transpose(encode)),encode/(n-1))
# 协方差矩阵特征值分解用于计算马氏距离
lambdas,v=np.linalg.eig(S)
# 低维度空间的马氏距离(方便起见仍用T2表示)
T2 = np.array([xi.dot(np.diag(lambdas**-1)).dot(xi.T) for xi in encode])
print(T2.size)
# 得到的T2为一维数组，resize为二维数组
T2 = T2.reshape(T2.size,1)
# print(T2)
kde=KernelDensity(kernel='gaussian',bandwidth=0.5).fit(T2)
print(T2[:3])

# print(enumerate(train_loader))
# for epoch in range(Epochs):
#     for  x , _ in enumerate(train_loader):
#         loss_total=0
#         encoded,decoded=model(x)
#         loss=loss_fun(decoded,x)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_total += loss.item()
#     # compute the epoch training loss
#     loss = loss / len(train_loader)
#     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, Epochs, loss))
