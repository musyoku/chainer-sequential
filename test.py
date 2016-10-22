import numpy as np
from chainer import Variable
from sequential import Sequential
import link
import function
import util

# Linear test
x = np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32)
x = Variable(x)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(link.Linear(28*28, 500))
seq.add(link.BatchNormalization(500))
seq.add(link.Linear(500, 500))
seq.add(function.Activation("clipped_relu"))
seq.add(link.Linear(500, 500, use_weightnorm=True))
seq.add(function.Activation("crelu"))	# crelu outputs 2x 
seq.add(link.BatchNormalization(1000))
seq.add(link.Linear(1000, 500))
seq.add(function.Activation("elu"))
seq.add(link.Linear(500, 500, use_weightnorm=True))
seq.add(function.Activation("hard_sigmoid"))
seq.add(link.BatchNormalization(500))
seq.add(link.Linear(500, 500))
seq.add(function.Activation("leaky_relu"))
seq.add(link.Linear(500, 500, use_weightnorm=True))
seq.add(function.Activation("relu"))
seq.add(link.BatchNormalization(500))
seq.add(link.Linear(500, 500))
seq.add(function.Activation("sigmoid"))
seq.add(link.Linear(500, 500, use_weightnorm=True))
seq.add(function.Activation("softmax"))
seq.add(link.BatchNormalization(500))
seq.add(link.Linear(500, 500))
seq.add(function.Activation("softplus"))
seq.add(link.Linear(500, 500, use_weightnorm=True))
seq.add(function.Activation("tanh"))
seq.add(link.Linear(500, 10))
seq.build()

y = seq(x)
print y.data

# Conv test
x = np.random.normal(scale=1, size=(2, 3, 96, 96)).astype(np.float32)
x = Variable(x)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(link.Convolution2D(3, 64, ksize=4, stride=2, pad=0))
seq.add(link.BatchNormalization(64))
seq.add(function.Activation("relu"))
seq.add(link.Convolution2D(64, 128, ksize=4, stride=2, pad=0, use_weightnorm=True))
seq.add(link.BatchNormalization(128))
seq.add(function.Activation("relu"))
seq.add(link.Convolution2D(128, 256, ksize=4, stride=2, pad=0))
seq.add(link.BatchNormalization(256))
seq.add(function.Activation("relu"))
seq.add(link.Convolution2D(256, 512, ksize=4, stride=2, pad=0, use_weightnorm=True))
seq.add(link.BatchNormalization(512))
seq.add(function.Activation("relu"))
seq.add(link.Convolution2D(512, 1024, ksize=4, stride=2, pad=0))
seq.add(link.BatchNormalization(1024))
seq.add(function.Activation("relu"))
seq.add(function.reshape_1d())
seq.add(link.Linear(None, 10, use_weightnorm=True))
seq.add(function.softmax())
seq.build()
print seq.to_json()

y = seq(x)
print y.data

# Deconv test
x = np.random.normal(scale=1, size=(2, 128)).astype(np.float32)
x = Variable(x)

# compute required paddings
paddings = []


seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(link.Convolution2D(3, 64, ksize=4, stride=2, pad=9))
seq.add(link.BatchNormalization(64))
out_size.append(util.get_conv_outsize(16, 3, 7, 9))
seq.add(link.Convolution2D(64, 128, ksize=4, stride=2, pad=4, use_weightnorm=True))
seq.add(link.BatchNormalization(128))
out_size.append(util.get_conv_outsize(out_size[-1], 4, 1, 4))
seq.add(link.Convolution2D(128, 256, ksize=4, stride=2, pad=2))
seq.add(link.BatchNormalization(256))
out_size.append(util.get_conv_outsize(out_size[-1], 5, 4, 2))
seq.add(link.Convolution2D(256, 512, ksize=4, stride=2, pad=2))
seq.add(link.BatchNormalization(512))
out_size.append(util.get_conv_outsize(out_size[-1], 5, 4, 2))
seq.add(function.reshape((-1, out_size[-1] ** 2 * 512)))
seq.build()
