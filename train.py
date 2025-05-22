import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
import time
import os
import math
import json 
import argparse
import numpy as np
from models import *
import shutil
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
model_list=["MLP", "Transformer", "Resnet", "GeoSigNet"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="dataset name (csv file)")
parser.add_argument("--num_encoder_layer", type=int,  default=0, help="number of encoder layer")
parser.add_argument("--resnet_version", type=int, default=50, help="resnet version")
parser.add_argument("--tag", type=str,  default="test", help="tag")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size",  type=int, default=1024, help="batch size")
parser.add_argument("--warmup_epochs", type=int, default=5, help="number of warmup_epochs")
parser.add_argument("--warmup_factor",  type=int, default=10, help="warmup_factor")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--test_size", type=float, default=0.1, help="test size")
parser.add_argument("--finetune_flag", type=bool, default=False, help="finetune flag")
parser.add_argument("--model_path", type=str, default="", help="finetune model path")
parser.add_argument("--model_name", type=str, default=None, help="model name")
parser.add_argument("--add_columns", type=str, default=None, help="add columns")

opt = parser.parse_args()

dataset_name=opt.dataset
num_epochs = opt.epochs
init_lr=opt.lr
batch_size=opt.batch_size
test_size=opt.test_size
finetune_flag=opt.finetune_flag
model_path=opt.model_path
num_encoder_layers=opt.num_encoder_layer
resnet_version=opt.resnet_version
model_name=opt.model_name
add_columns=opt.add_columns
if model_name=="Transformer":
    dim_feedforward=256
    nhead=4
    tag=f"{opt.tag}_layer{num_encoder_layers}_epoch{num_epochs}"
elif model_name=="Resnet":
    tag=f"{opt.tag}_resnet{resnet_version}_epoch{num_epochs}"
else:
    tag=f"{opt.tag}_epoch{num_epochs}"


if dataset_name==None:
    print("Please check dataset setting.")
    exit()
if model_name not in model_list:
    print(f"{model_name} is not in model list, close the program.")
    exit()



if not os.path.exists(model_name):
    os.makedirs(model_name)

log_folder=f"{model_name}/{tag}"
if  os.path.exists(log_folder):
    print(f"warning: {log_folder} is exist, please check")
    exit()


if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_path=f"{log_folder}/train_log_{tag}.txt"

df = pd.read_csv(dataset_name)
print(f"origin dataset size: {len(df)}")
df = df[df["Path_loss"] != float("inf")]
print(f"dataset size that path loss !=inf:{len(df)}")
# 分割資料集
# X = df.drop(['Received_power', 'Frequency_GHz', 'LOS_Flag'], axis=1)
add_columns_list=[]
if add_columns!=None:
    add_columns_list=add_columns.split(",")

input_columns=["Frequency_GHz"]
if "GPS" in add_columns_list:
    GPS_columns=["Tx_x", "Tx_y", "Tx_z", "Rx_x", "Rx_y", "Rx_z"]
    input_columns=GPS_columns+input_columns
    add_columns_list.remove("GPS")

if "Rx_z" in input_columns:
    # 找到 "Rx_z" 的索引位置
    index = input_columns.index("Rx_z")
    # 在 "Rx_z" 之後、"Frequency" 之前插入 add_columns_list
    input_columns = input_columns[:index+1] + add_columns_list + input_columns[index+1:]
else:
    input_columns = add_columns_list + input_columns

# Tx_x,Tx_y,Tx_z,Rx_x,Rx_y,Rx_z,Distance,Angle,Frequency_GHz,LOS_Flag,Path_loss

input_data = df[input_columns]
output_data = df["Path_loss"]
input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size=test_size, random_state=42)

# 标准化
scaler = StandardScaler()
input_data_scaled=scaler.fit_transform(input_data)
input_data_train_scaled = scaler.transform(input_data_train)
input_data_test_scaled = scaler.transform(input_data_test)
dump(scaler, f'{log_folder}/scaler_{tag}.joblib')




class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# 添加checkpoint相關的函數
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(filename):
    if os.path.isfile(filename):
        print(f"=> 加載checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"=> 沒有找到checkpoint '{filename}'")
        return None


# 创建数据加载器
train_dataset = CustomDataset(input_data_train_scaled, output_data_train.to_numpy())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(input_data_test_scaled, output_data_test.to_numpy())
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


input_dim=len(input_data.columns)
output_dim=1
dropout=0.1
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


# model_list=["DNN", "Transformer", "Resnet", "my_model"]
if model_name=="MLP":
    model = MLP(input_dim=input_dim, output_dim=output_dim)
elif model_name=="Transformer":
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
elif model_name=="Resnet":
    model = CustomResNet(input_features=input_dim, output_dim=output_dim, resnet_version=resnet_version)
elif model_name=="GeoSigNet":
    if ("Distance" in add_columns) and ("Angle" not in add_columns) and ("LOS_Flag" not in add_columns) and ("GPS" not in add_columns):
        model=GeoSigNet_Distance(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
    elif ("Distance" in add_columns) and ("Angle" in add_columns) and ("LOS_Flag" not in add_columns) and ("GPS" not in add_columns):
        model = GeoSigNet_DistanceAngle(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
    elif ("Distance" in add_columns) and ("Angle" not in add_columns) and ("LOS_Flag" in add_columns) and ("GPS" not in add_columns):
        model = GeoSigNet_DistanceLOS(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
    elif ("Distance" in add_columns) and ("Angle" not in add_columns) and ("LOS_Flag" not in add_columns) and ("GPS" in add_columns):
        model = GeoSigNet_DistanceGPS(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
    else:
        model = GeoSigNet(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
else:
    print("model does not exist")
    print("exit...")
    exit()


        

# model = TransformerModel(input_dim=input_dim, output_dim=output_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)


if finetune_flag: 
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        print("已成功load原始模型")
    else:
        print("未找到需finetune的模型")

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr)
max_iter=math.ceil(input_data_train.shape[0]/batch_size)*num_epochs


warmup_epochs = opt.warmup_epochs  
warmup_factor = opt.warmup_factor  


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return warmup_factor
    else:
        return 1.0

scheduler1 = LambdaLR(optimizer, lr_lambda)
scheduler2 = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

log_config={
    "dataset_name": dataset_name,
    "num_epochs": num_epochs,
    "finetune_flag": finetune_flag,
    "load_model_path": model_path,
    "init_lr": init_lr,
    "batch_size":batch_size,
    "test_size":test_size,
    "train dataset size":input_data_train.shape[0],
    "validation_dataset_size":input_data_test.shape[0],
    "validation_rate":test_size,
    "input_dim":input_dim,
    "input_columns":str(input_data_train.columns.values),
    "output_dim" : output_dim, 
    "dropout" : dropout,
    "total_params": total_params,
}
if model_name=="Transformer":
    temp_dict={    
        "nhead" : nhead, 
        "num_encoder_layers" : num_encoder_layers,  
        "dim_feedforward" : dim_feedforward
        }
    log_config.update(temp_dict)
elif model_name=="Resnet":
    temp_dict={    
        "resnet_version" : resnet_version, 
        }
    log_config.update(temp_dict)

print(f"hyper parameter:\n{log_config}")

with open(log_path, "w") as file:
    file.write(f"{json.dumps(log_config)}\n")
    pass

best_mse=float("inf")
start_time=time.time()

# 檢查是否有checkpoint可以恢復
start_epoch = 0
best_mse = float('inf')
if os.path.exists(f"{log_folder}/checkpoint.pth.tar"):
    checkpoint = load_checkpoint(f"{log_folder}/checkpoint.pth.tar")
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_mse = checkpoint['best_mse']
        print(f"=> 從epoch {start_epoch}恢復訓練")
# 訓練循環
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
    scheduler.step()
    # 驗證階段
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            validation_loss += loss.item()

    train_loss_avg = running_loss / len(train_loader)
    validation_loss_avg = validation_loss / len(test_loader)
    
    print(f"Epoch {epoch+1}, Train loss: {train_loss_avg:.4f}, Validation loss: {validation_loss_avg:.4f}")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Current learning rate: {current_lr}")
    with open(log_path, "a+") as file:
        file.write(f"Epoch {epoch+1}, Train loss: {train_loss_avg:.4f}, Validation loss: {validation_loss_avg:.4f}\n")
    
    # 保存checkpoint
    is_best = validation_loss_avg < best_mse
    best_mse = min(validation_loss_avg, best_mse)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_mse': best_mse,
        'optimizer': optimizer.state_dict(),
    }, is_best, filename=f"{log_folder}/checkpoint.pth.tar")
    
    if is_best:
        torch.save(model, f"{log_folder}/best_model_{tag}.pth")
        print(f"epoch:{epoch+1}, save best_model_{tag}.pth")
    
    if (epoch+1) % 10 == 0:
        torch.save(model, f"{log_folder}/lastest_model_{tag}.pth")