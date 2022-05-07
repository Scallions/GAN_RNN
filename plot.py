from time import time
import pandas as pd
# import sys
# sys.path.append("..")
# print(sys.path)
from model.informer.informer import Informer
import paddle
import matplotlib.pyplot as plt

data_dir = "/Volumes/HDD/Data/ztd/"

# 加载数据
df = pd.read_csv(data_dir + "all_r/remove_trend.csv", parse_dates=['time'])

# 加载数据集
from model.informer.data.data_loader import Dataset_hour
ds = Dataset_hour(".", flag="val")

model_dir = "/Volumes/HDD/Data/ztd/gan/best_infor.pdmodel"
model = Informer(64,64,64,384,96,96,attn="full")
model.set_state_dict(paddle.load(model_dir))
model.eval()

# from model.rnn.mlstm import Seq2seq
# seq = Seq2seq(35,35,512)
# seq.set_state_dict(paddle.load("out_seq/best1.pdmodel"))
# seq.eval()

start = df.shape[0] - 1*13*30*24 - 16*24
from datetime import datetime, timedelta
start_t = df.time[0] + timedelta(hours=start)
print(start_t)
# 2020-5-5
### 预测开始时间
startpp = datetime(2019, 3, 1)
startp = startpp - timedelta(hours=16*24)
print(startpp)

# load era
import xarray as xr
import numpy as np
eras = xr.load_dataarray(data_dir + "pwv.nc")
# era = eras.sel(time=slice(start_t, start_t + timedelta(hours=8*30*24)))
era_max = eras.min(axis=0)[:31,:64].values
era_min = eras.max(axis=0)[:31,:64].values

# pstart_t = datetime
# 预测
idx = 0
didx = int((startp-start_t).total_seconds()/3600)

# informer
[x, y, x_date, y_date] = map(lambda x: paddle.to_tensor(x).astype("float32").unsqueeze(0), ds[didx])
y_t = y[:,-96:,:]
dec_inp = paddle.zeros([1,96,64])
dec_inp = paddle.concat([y[:,:96,:], dec_inp], axis=1)
y_p = model(x,x_date,dec_inp,y_date)
y_p = ds.inverse_transform(y_p)
x_r = paddle.concat([x, y_t], axis=1)
x_r = ds.inverse_transform(x_r)

pstart = startpp
pend = startpp + timedelta(hours=4*24-1)
tidx = pd.date_range(start=pstart,end=pend, freq='H')

days = (tidx-datetime(2010,1,1)).total_seconds()/86400
length = len(days)
x = days.values.reshape((length, 1))
sinx = np.sin(x * np.pi * 2 / 365.25)
cosx = np.cos(x * np.pi * 2 / 365.25)
sin2x = np.sin(2 * x * np.pi * 2 / 365.25)
cos2x = np.cos(2 * x * np.pi * 2 / 365.25)
ones = np.ones((length, 1))
data = np.hstack((ones, x, sinx, cosx, sin2x, cos2x))
trend = pd.read_csv(data_dir + "all_r/trends.csv").values[:,1:].T
trends = np.dot(data, trend).reshape(1,96,64)
y_p = y_p + paddle.to_tensor(trends.astype(float))

# plt.figure()
# plt.plot(x_r[0,:,0], label="true")
# plt.plot(list(range(384,384+96)), y_p[0,:,0], label="pred")
# plt.show()
# plt.close()

# start = df.shape[0]-360*24 + didx
# tidxs = df["time"][start:start+20*24]
# y_all = paddle.concat([x[0,:,:], y[0,-96:,:]], axis=0)
# y_all = ds.inverse_transform(y_all)
# y_all = y_all[:,idx]
# fig = plt.figure(dpi=300)
# fig.set_figwidth(5)
# plt.plot(tidxs, y_all, label="raw")
# plt.plot(tidxs[-96:], y_p[0,-96:,idx], label="transformer")
# fig.autofmt_xdate()
# plt.legend()
# plt.show()

## gan
from model.gan.generators.dcgenerator import DCGenerator
gan = DCGenerator(64,1)
gan.set_state_dict(paddle.load(data_dir+"gan/G_latest.pdmodel"))
gan.eval()

out = gan(paddle.to_tensor((y_p[0,:,:])-5.78)/3.43)

import numpy as np
from cartopy.crs import LambertAzimuthalEqualArea, PlateCarree
# import matplotlib.pyplot as plt
import matplotlib.path as mpath
proj = LambertAzimuthalEqualArea(-40, 60)
# ax = plt.axes(projection=proj)
# ax = plt.axes(projection=PlateCarree())
# print(type(ax))
# ax.set_global()
# ax.set_extent([-75,-12,55,85])
# ax.set_xlim(-75, -12)
# ax.set_ylim(55, 85)
# ax.set_boundary(mpath.Path(np.array([[-75,55],[-75,85],[-12,85],[-12,55], [-75,55]]), _interpolation_steps=100), transform=PlateCarree())
# ax.coastlines()
# ax.gridlines(draw_labels=True)
# x = np.arange(-75, -11)
# y = np.arange(55, 86)
# xs, ys = np.meshgrid(x,y)
# mesh = ax.pcolormesh(xs, ys, out[0,:,:], transform=PlateCarree())
# plt.show()


def to_realscale(pwvs):
    return pwvs.detach().numpy()*(era_max-era_min)+era_min

out = to_realscale(out)
out = out[:,::-1,:]

import gif
gif.options.matplotlib["dpi"] = 300

timel = 96
vmin = max(0, np.min(out[:timel,:,:]))
vmax = np.max(out[:timel,:,:])
import matplotlib as mpl
cmap = mpl.colormaps["jet"]

rmse = []
@gif.frame
def plot(pidx):
    # plt.title(f"{pidx}")
    cdata = startp + timedelta(hours=pidx+16*24)
    fig, axs = plt.subplots(1,2, subplot_kw={'projection': PlateCarree()})
    ax = axs[0]
    ax.set_title(f"GPS:{cdata}")
    ax.set_extent([-75,-12,55,85])
    ax.coastlines()
    # ax.gridlines(draw_labels=True)
    x = np.arange(-75, -11)
    y = np.arange(55, 86)
    xs, ys = np.meshgrid(x,y)
    mesh = ax.pcolormesh(xs, ys, out[pidx,:,:], vmin=vmin, vmax= vmax, transform=PlateCarree(), cmap = cmap)
    # plt.imshow(out[pidx,:,:])
    # plt.axis('off')
    ax = axs[1]
    ax.set_title(f"ERA:{cdata}")
    ax.set_extent([-75,-12,55,85])
    ax.coastlines()
    # ax.gridlines(draw_labels=True)
    x = np.arange(-75, -11)
    y = np.arange(55, 86)
    xs, ys = np.meshgrid(x,y)
    era = eras.sel(time=cdata).values[:31,:64][::-1,:]
    erass.append(era)
    mesh = ax.pcolormesh(xs, ys, era, vmin=vmin, vmax=vmax, transform=PlateCarree(), cmap=cmap)
    # plt.colorbar(ax=ax)
    fig.colorbar(mesh, ax=axs, orientation='horizontal')
    # plt.tight_layout()
    rmse.append(np.sqrt(np.mean((out[pidx,:,:]-era)**2)))

frames = []
erass = []
for i in range(timel):
    frame = plot(i)
    frames.append(frame)

np.save("eras.npy", np.array(erass))
np.save("outs.npy", out[:timel,:,:])

gif.save(frames, f"{startpp}.gif", duration=5, unit="s", between="startend")
print(np.mean(rmse))