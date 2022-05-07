import pandas as pd
# import sys
# sys.path.append("..")
# print(sys.path)
from model.informer.informer import Informer
import paddle
import matplotlib.pyplot as plt

data_dir = "/Volumes/HDD/Data/ztd/"

# 加载数据
df = pd.read_csv(data_dir + "pwv_1h_hz_grl_filter_miss.csv", parse_dates=['time'])

# 加载数据集
from model.informer.data.data_loader import Dataset_hour
ds = Dataset_hour(".", flag="val")

model_dir = "/Volumes/HDD/Data/ztd/informer_best1.pdmodel"

model = Informer(35,35,35,384,96,96,attn="full")
model.set_state_dict(paddle.load(model_dir))
model.eval()


start = df.shape[0] - 8*30*24 - 12*30*24 - 16*24
from datetime import datetime, timedelta
start_t = df.time[0] + timedelta(hours=start)
# 2020-5-5
startp = datetime(2020, 6, 26)

# pstart_t = datetime
# 预测
# idx = 33
# didx = int((startp-start_t).total_seconds()/3600)

# # informer
# [x, y, x_date, y_date] = map(lambda x: paddle.to_tensor(x).astype("float32").unsqueeze(0), ds[didx])
# y_t = y[:,-96:,:]
# dec_inp = paddle.zeros([1,96,35])
# dec_inp = paddle.concat([y[:,:96,:], dec_inp], axis=1)
# y_p = model(x,x_date,dec_inp,y_date)
# y_p = ds.inverse_transform(y_p)
# x_r = paddle.concat([x, y_t], axis=1)
# x_r = ds.inverse_transform(x_r)

# plt.figure()
# plt.plot(x_r[0,:,idx], label="true")
# plt.plot(list(range(384,384+96)), y_p[0,:,idx], label="pred")
# plt.show()
# plt.close()


## 内插

ds1 = Dataset_hour(".", flag="train")
idx = 33
didx = 11
# didx = int((startp-start_t).total_seconds()/3600)

# informer
[x, y, x_date, y_date] = map(lambda x: paddle.to_tensor(x).astype("float32").unsqueeze(0), ds1[didx])
y_t = y[:,-96:,:]
dec_inp = paddle.zeros([1,96,35])
dec_inp = paddle.concat([y[:,:96,:], dec_inp], axis=1)
y_p = model(x,x_date,dec_inp,y_date)
y_p = ds1.inverse_transform(y_p)
x_r = paddle.concat([x, y_t], axis=1)
x_r = ds1.inverse_transform(x_r)

plt.figure()
plt.plot(x_r[0,:,idx], label="true")
plt.plot(list(range(384,384+96)), y_p[0,:,idx], label="pred")
plt.show()
plt.close()