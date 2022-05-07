

from time import time
from model.skillful_nowcasting.models.modules import layers
from model.skillful_nowcasting.models.modules import laten_stack
from model.skillful_nowcasting.models import discriminators
from model.skillful_nowcasting.models import generators

import paddle

# inp = paddle.rand([2,3,10,10])
# out = layers.downsample_avg_pool(inp)
# print(out.shape)

# inp = paddle.rand([2,3,4,4,4])
# out = layers.downsample_avg_pool3d(inp)
# print(out.shape)

# inp = paddle.rand([2,1,4,4])
# out = layers.Conv2D(1,3,3)(inp)
# print(out.shape)

# inp = paddle.rand([2,3,4,4])
# out = layers.SNConv2D(3,2,3)(inp)
# print(out.shape)

# inp = paddle.rand([2,3,4,4,4])
# out = layers.SNConv3D(3,2,3)(inp)
# print(out.shape)

# inp = paddle.rand([3,2,4,4])
# out = layers.ApplyAlongAxis(paddle.add, 1)(inp, inp)
# print(out.shape)

# out = laten_stack.LatentCondStack()(2)
# print(out.shape) # 2 768 8 8

# inp = paddle.rand([2,1,256,256])
# out = discriminators.DBlock(1,3,3)(inp)
# print(out.shape)

# inp = paddle.rand([2,3,1,128,128])
# out = discriminators.SpatialDiscriminator(1)(inp)
# print(out.shape)

# inp = paddle.rand([2,24,1,128,128])
# out = discriminators.TemporalDiscriminator(1)(inp)
# print(out.shape)

inp = paddle.rand([2,22,1,256,256])
out = discriminators.Discriminator(1)(inp)
print(out.shape)

# inp = paddle.rand([2,1,4,4])
# out = generators.GBlock(1,3,3)(inp)
# print(out.shape)

# inp = paddle.rand([3,2,4,4])
# out = generators.ConvGRU(2,2,3)(inp, inp)
# print(out.shape)

# inp = paddle.rand([3,2,4,4])
# out = generators.UpsampleGBlock(2,2,3)(inp)
# print(out.shape)


# inp = paddle.rand([1,2,1,256,256])
# out = generators.ConditioningStack(1,2)(inp)
# print(out[0].shape)

# out = generators.Sampler()(out)

# inp = paddle.rand([1,2,1,256,256])
# out = generators.Generator(num_channels=1,lead_time=2, time_delta=1)(inp)
# print(out.shape)