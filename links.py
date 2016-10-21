import numpy
import chainer
import links

class Link(Object):

	def __init__(self):
		self.classname = "Link"

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

	def to_link(self):
		raise NotImplementedError()

	def dump(self):
		print "Link: {}".format(self.name)
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

class Bias(Link):
	def __init__(self, axis=1, shape=None):
		super(Bias, self).__init__()
		self.name = "Bias"
		self.axis = axis
		self.shape = shape

	def to_link(self):
		args = self.to_dict()
		return chainer.links.Bias(**args)

class Bilinear(Link):
	def __init__(self, left_size, right_size, out_size, nobias=False):
		super(Bilinear, self).__init__()
		self.name = "Bilinear"
		self.left_size = left_size
		self.right_size = right_size
		self.out_size = out_size
		self.nobias = nobias

	def to_link(self):
		args = self.to_dict()
		return chainer.links.Bilinear(**args)


class Convolution2D(Link):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, use_weightnorm=False):
		super(Convolution2D, self).__init__()
		self.name = "Convolution2D"
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias
		self.use_cudnn = use_cudnn
		self.use_weightnorm = use_weightnorm

	def to_link(self):
		args = self.to_dict()
		if self.use_weightnorm:
			return links.weightnorm.Convolution2D(**args)
		return chainer.links.Convolution2D(**args)

class Deconvolution2D(Link):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, outsize=None, use_cudnn=True, use_weightnorm=False):
		super(Deconvolution2D, self).__init__()
		self.name = "Deconvolution2D"
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
		self.use_weightnorm = use_weightnorm

	def to_link(self):
		args = self.to_dict()
		if self.use_weightnorm:
			return links.weightnorm.Deconvolution2D(**args)
		return chainer.links.Deconvolution2D(**args)

class DilatedConvolution2D(Link):
	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, wscale=1, bias=0, nobias=False, use_cudnn=True):
		super(DilatedConvolution2D, self).__init__()
		self.name = "DilatedConvolution2D"
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

	def to_link(self):
		args = self.to_dict()
		return chainer.links.DilatedConvolution2D(**args)

class EmbedID(Link):
	def __init__(self, in_size, out_size, ignore_label=None):
		super(EmbedID, self).__init__()
		self.name = "EmbedID"
		self.in_size = in_size
		self.out_size = out_size
		self.ignore_label = ignore_label

	def to_link(self):
		args = self.to_dict()
		return chainer.links.EmbedID(**args)

class GRU(Link):
	def __init__(self, n_units, n_inputs=None):
		super(GRU, self).__init__()
		self.name = "GRU"
		self.n_units = n_units
		self.n_inputs = n_inputs

	def to_link(self):
		args = self.to_dict()
		return chainer.links.GRU(**args)

class Inception(Link):
	def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
		super(Inception, self).__init__()
		self.name = "Inception"
		self.in_channels = in_channels
		self.out1 = out1
		self.proj3 = proj3
		self.out3 = out3
		self.proj5 = proj5
		self.out5 = out5
		self.proj_pool = proj_pool

	def to_link(self):
		args = self.to_dict()
		return chainer.links.Inception(**args)

class Linear(Link):
	def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False, use_weightnorm=False):
		super(Linear, self).__init__()
		self.name = "Linear"
		self.in_size = in_size
		self.out_size = out_size
		self.wscale = wscale
		self.bias = bias
		self.nobias = nobias
		self.use_weightnorm = use_weightnorm

	def to_link(self):
		args = self.to_dict()
		if self.use_weightnorm:
			return links.weightnorm.Linear(**args)
		return chainer.links.Linear(**args)

class LSTM(Link):
	def __init__(self, in_size, out_size):
		super(LSTM, self).__init__()
		self.name = "LSTM"
		self.in_size = in_size
		self.out_size = out_size

	def to_link(self):
		args = self.to_dict()
		return chainer.links.LSTM(**args)

class StatelessLSTM(Link):
	def __init__(self, in_size, out_size):
		super(StatelessLSTM, self).__init__()
		self.name = "StatelessLSTM"
		self.in_size = in_size
		self.out_size = out_size

	def to_link(self):
		args = self.to_dict()
		return chainer.links.StatelessLSTM(**args)

class Scale(Link):
	def __init__(self, axis=1, W_shape=None, bias_term=False, bias_shape=None):
		super(Scale, self).__init__()
		self.name = "Scale"
		self.axis = axis
		self.W_shape = W_shape
		self.bias_term = bias_term
		self.bias_shape = bias_shape

	def to_link(self):
		args = self.to_dict()
		return chainer.links.Scale(**args)

class StatefulGRU(Link):
	def __init__(self, in_size, out_size, bias_init=0):
		super(StatefulGRU, self).__init__()
		self.name = "StatefulGRU"
		self.in_size = in_size
		self.out_size = out_size
		self.bias_init = bias_init

	def to_link(self):
		args = self.to_dict()
		return chainer.links.StatefulGRU(**args)

class StatefulPeepholeLSTM(Link):
	def __init__(self, in_size, out_size):
		super(StatefulPeepholeLSTM, self).__init__()
		self.name = "StatefulPeepholeLSTM"
		self.in_size = in_size
		self.out_size = out_size

	def to_link(self):
		args = self.to_dict()
		return chainer.links.StatefulPeepholeLSTM(**args)

class BatchNormalization(Link):
	def __init__(self, size, decay=0.9, eps=2e-05, dtype=numpy.float32, use_gamma=True, use_beta=True, use_cudnn=True):
		super(BatchNormalization, self).__init__()
		self.name = "BatchNormalization"
		self.size = size
		self.decay = decay
		self.eps = eps
		self.dtype = dtype
		self.use_gamma = use_gamma
		self.use_beta = use_beta
		self.use_cudnn = use_cudnn

	def to_link(self):
		args = self.to_dict()
		return chainer.links.BatchNormalization(**args)
