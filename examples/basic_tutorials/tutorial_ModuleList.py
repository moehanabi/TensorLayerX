#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'paddle'

from tensorlayerx.nn import Module, ModuleList, Linear
import tensorlayerx as tlx

d1 = Linear(out_features=800, act=tlx.ReLU, in_features=784, name='linear1')
d2 = Linear(out_features=800, act=tlx.ReLU, in_features=800, name='linear2')
d3 = Linear(out_features=10, act=tlx.ReLU, in_features=800, name='linear3')

layer_list = ModuleList([d1, d2])
# Inserts a given d2 before a given index in the list
layer_list.insert(1, d2)
layer_list.insert(2, d2)
# Appends d2 from a Python iterable to the end of the list.
layer_list.extend([d2])
# Appends a given d3 to the end of the list.
layer_list.append(d3)

print(layer_list)


class model(Module):

    def __init__(self):
        super(model, self).__init__()
        self._list = layer_list

    def forward(self, inputs):
        output = self._list[0](inputs)
        for i in range(1, len(self._list)):
            output = self._list[i](output)
        return output


net = model()
print(net.trainable_weights)
print(net)
print(net(tlx.nn.Input((10, 784))))