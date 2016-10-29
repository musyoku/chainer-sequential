import numpy as np
from chainer import *
from minibatch_discrimination import * 

x = Variable(np.random.uniform(size=(3, 4)).astype(np.float32))
layer = MinibatchDiscrimination(4, 2, 4)
y = layer(x)