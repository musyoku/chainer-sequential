import numpy as np
from chainer import *
from minibatch_discrimination import * 

x = Variable(np.ones((2, 4), dtype=np.float32))
layer = MinibatchDiscrimination(4, 2, 3)
y = layer(x)