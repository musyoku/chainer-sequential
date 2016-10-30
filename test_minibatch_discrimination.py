from chainer import Variable
import numpy as np
from minibatch_discrimination import MinibatchDiscrimination

x = Variable(np.random.normal(size=(4, 10)).astype(np.float32))
layer = MinibatchDiscrimination(10, 5, 30)
print layer(x).data