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



# 预测
idx = 0
didx = 0

# informer
[x, y, x_date, y_date] = map(lambda x: paddle.to_tensor(x).astype("float32").unsqueeze(0), ds[didx])
y_t = y[:,-96:,:]
dec_inp = paddle.zeros([1,96,35])
dec_inp = paddle.concat([y[:,:96,:], dec_inp], axis=1)
y_p = model(x,x_date,dec_inp,y_date)
y_p = ds.inverse_transform(y_p)

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
from model.gan.generators.pwvgan import DCGenerator
gan = DCGenerator(35,1)
gan.set_state_dict(paddle.load(data_dir+"G_best.pdmodel"))
gan.eval()

out = gan(paddle.to_tensor((y_p[0,:,:])-5.78)/3.43)

import numpy as np
from cartopy.crs import LambertAzimuthalEqualArea, PlateCarree
# import matplotlib.pyplot as plt
import matplotlib.path as mpath
proj = LambertAzimuthalEqualArea(-40, 60)
ax = plt.axes(projection=proj)
print(type(ax))
# ax.set_global()
ax.set_extent([-75,-12,55,85])
# ax.set_xlim(-75, -12)
# ax.set_ylim(55, 85)
# ax.set_boundary(mpath.Path(np.array([[-75,55],[-75,85],[-12,85],[-12,55], [-75,55]]), _interpolation_steps=100), transform=PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
x = np.arange(-75, -11)
y = np.arange(55, 86)
xs, ys = np.meshgrid(x,y)
mesh = ax.pcolormesh(xs, ys, out[0,:,:], transform=PlateCarree())
plt.show()


# import gif
# gif.options.matplotlib["dpi"] = 300

# @gif.frame
# def plot(pidx):
#     plt.title(f"{pidx}")
#     plt.imshow(out[pidx,:,:])
#     plt.axis('off')
#     plt.tight_layout()

# frames = []
# for i in range(96):
#     frame = plot(i)
#     frames.append(frame)

# gif.save(frames, "test.gif", duration=5, unit="s", between="startend")