"""
gan 接口类
"""

import abc
import paddle.nn as nn


class IGAN(abc.ABC):


    @abc.abstractmethod
    def forward(self):
        pass


    @abc.abstractmethod
    def train_gan(self):
        pass

    def set_requires_grad(self, nets, requires_grad: bool=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Args:
            nets (network list): a list of networks
            requires_grad (bool): whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad