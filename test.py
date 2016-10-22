import numpy as np
from chainer import Variable
from sequential import Sequential
import link
import function

x = np.random.normal(scale=1, size=(128, 28*28))
x = Variable(x)

seq = Sequential()
seq.add(link.Linear(1, 1))
seq.add(link.BatchNormalization(500))
seq.add(function.relu())
seq.add(link.Linear(1, 1))
seq.weight_initializer = "GlorotNormal"
seq.weight_init_std = 0.05
seq.build()

y = seq(x)
print y.data