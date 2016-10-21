import numpy
import layer
from chainer import links as L
import weightnorm as WN

class Bias(layer.layer):
	def __init__(self, axis=1, shape=None):
		self.axis = axis
		self.shape = shape

class Bilinear(layer.layer):
	def __init__(self, left_size, right_size, out_size, nobias=False):
		self.left_size = left_size
		self.right_size = right_size
		self.out_size = out_size
		self.nobias = nobias
		self._link = L.Bilinear

class Convolution2D(layer.layer):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias
		self.use_cudnn = use_cudnn
		self._link = L.Convolution2D

class WeightnormConvolution2D(Convolution2D):
	pass

class Deconvolution2D(layer.layer):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, outsize=None, use_cudnn=True):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias
		self.outsize = outsize
		self.use_cudnn = use_cudnn

class WeightnormDeconvolution2D(Deconvolution2D):
	pass

class DilatedConvolution2D(layer.layer):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, wscale=1, bias=0, nobias=False, use_cudnn=True):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.dilate = dilate
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias
		self.use_cudnn = use_cudnn

class EmbedID(layer.layer):
	def __init__(self, in_size, out_size, ignore_label=None):
		self.in_size = in_size
		self.out_size = out_size
		self.ignore_label = ignore_label

class GRU(layer.layer):
	def __init__(self, n_units, n_inputs=None):
		self.n_units = n_units
		self.n_inputs = n_inputs

class Inception(layer.layer):
	def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
		self.in_channels = in_channels
		self.out1 = out1
		self.proj3 = proj3
		self.out3 = out3
		self.proj5 = proj5
		self.out5 = out5
		self.proj_pool = proj_pool

class Linear(layer.layer):
	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False):
		self.in_size = in_size
		self.out_size = out_size
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias

class WeightnormLinear(Linear):
	pass

class LSTM(layer.layer):
	def __init__(self, in_size, out_size):
		self.in_size = in_size
		self.out_size = out_size

class StatelessLSTM(layer.layer):
	def __init__(self, in_size, out_size):
		self.in_size = in_size
		self.out_size = out_size

class Scale(layer.layer):
	def __init__(self, axis=1, W_shape=None, bias_term=False, bias_shape=None):
		self.axis = axis
		self.W_shape = W_shape
		self.bias_term = bias_term
		self.bias_shape = bias_shape

class StatefulGRU(layer.layer):
	def __init__(self, in_size, out_size, bias_init=0):
		self.in_size = in_size
		self.out_size = out_size
		self.bias_init = bias_init

class StatefulPeepholeLSTM(layer.layer):
	def __init__(self, in_size, out_size):
		self.in_size = in_size
		self.out_size = out_size

class BatchNormalization(layer.layer):
	def __init__(self, size, decay=0.9, eps=2e-05, dtype=numpy.float32, use_gamma=True, use_beta=True, use_cudnn=True):
		self.size = size
		self.decay = decay
		self.eps = eps
		self.dtype = dtype
		self.use_gamma = use_gamma
		self.use_beta = use_beta
		self.use_cudnn = use_cudnn
