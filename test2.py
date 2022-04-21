from model.libs.config import Config
cfg = Config("test.yml")
# print(cfg.MODEL)
# print(cfg.informer)
# print(cfg.ds)



import paddle

val = cfg.val
vds = paddle.io.DataLoader(cfg.val, batch_size=2, use_shared_memory=False)

batch_x,batch_y,batch_x_mark,batch_y_mark = next(iter(vds))
batch_x = batch_x.astype("float32")
batch_y = batch_y.astype("float32")
batch_x_mark = batch_x_mark.astype("float32")
batch_y_mark = batch_y_mark.astype("float32")
std = batch_x.std(axis=2).mean(axis=1).unsqueeze(1).unsqueeze(1).tile([1,480,1])

bb = paddle.concat([batch_x, batch_y[:,-96:,:]], axis=1)
mark = paddle.concat([batch_x_mark, batch_y_mark[:,-96:,:]], axis=1)
bx = paddle.concat([bb, mark, std],axis=2)

model = cfg.tft

out = model(bx)

print(out.shape)