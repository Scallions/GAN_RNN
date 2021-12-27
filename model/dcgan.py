import paddle
import functools
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import BatchNorm2D



class Identity(nn.Layer):
    def forward(self, x):
        return x

class _SpectralNorm(nn.SpectralNorm):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps,
                                            dtype)

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(type="spectral_norm",
                               inputs=inputs,
                               outputs={
                                   "Out": out,
                               },
                               attrs={
                                   "dim": self._dim,
                                   "power_iters": _power_iters,
                                   "eps": self._eps,
                               })

        return out

class Spectralnorm(paddle.nn.Layer):
    def __init__(self, layer, dim=0, power_iters=1, eps=1e-12, dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters,
                                           eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape,
                                                 dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

def build_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm,
            param_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(1.0, 0.02)),
            bias_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0)),
            trainable_statistics=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(1.0),
                learning_rate=0.0,
                trainable=False),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0),
                                       learning_rate=0.0,
                                       trainable=False))
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


class DCDiscriminator(nn.Layer):
    """Defines a DCGAN discriminator"""
    def __init__(self, input_nc, ndf=64, norm_type='instance'):
        """Construct a DCGAN discriminator

        Parameters:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            norm_type (str): normalization layer type
        """
        super(DCDiscriminator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2D(input_nc,
                      ndf,
                      kernel_size=kw,
                      stride=2,
                      padding=padw,
                      bias_attr=use_bias),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        n_downsampling = 4

        # gradually increase the number of filters
        for n in range(1, n_downsampling):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm_type == 'batch':
                sequence += [
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    BatchNorm2D(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]
            else:
                sequence += [
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias_attr=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]

        nf_mult_prev = nf_mult

        # output 1 channel prediction map
        sequence += [
            nn.Conv2D(ndf * nf_mult_prev,
                      1,
                      kernel_size=kw,
                      stride=1,
                      padding=0)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
