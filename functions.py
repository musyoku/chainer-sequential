from chainer import functions as F

class Function():
	def __init__(self):
		self.classname = "Function"

	def __call__(self, x):
		raise NotImplementedError()

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

class Activation():
	def __init__(self, nonlinearity="relu"):
		self.nonlinearity = nonlinearity

	def to_function(self):
		if self.nonlinearity.lower() == "clipped_relu":
			return clipped_relu()
		if self.nonlinearity.lower() == "crelu":
			return crelu()
		if self.nonlinearity.lower() == "elu":
			return elu()
		if self.nonlinearity.lower() == "hard_sigmoid":
			return hard_sigmoid()
		if self.nonlinearity.lower() == "leaky_relu":
			return leaky_relu()
		if self.nonlinearity.lower() == "maxout":
			return maxout()
		if self.nonlinearity.lower() == "relu":
			return relu()
		if self.nonlinearity.lower() == "sigmoid":
			return sigmoid()
		if self.nonlinearity.lower() == "softmax":
			return softmax()
		if self.nonlinearity.lower() == "softplus":
			return softplus()
		if self.nonlinearity.lower() == "tanh":
			return tanh()
		raise NotImplementedError()

class clipped_relu(Function):
	def __init__(self, z=20.0):
		super(clipped_relu, self).__init__()
		self.z = z

	def __call__(self, x):
		return F.clipped_relu(x, self.z)

class crelu(Function):
	def __init__(self, axis=1):
		super(crelu, self).__init__()
		self.axis = axis

	def __call__(self, x):
		return F.crelu(x, self.axis)

class elu(Function):
	def __init__(self, alpha=1.0):
		super(elu, self).__init__()
		self.alpha = alpha

	def __call__(self, x):
		return F.elu(x, self.alpha)

class hard_sigmoid(Function):
	def __init__(self):
		super(hard_sigmoid, self).__init__()
		pass

	def __call__(self, x):
		return F.hard_sigmoid(x)

class leaky_relu(Function):
	def __init__(self, slope=0.2):
		super(leaky_relu, self).__init__()
		self.slope = slope

	def __call__(self, x):
		return F.leaky_relu(x, self.slope)

class log_softmax(Function):
	def __init__(self, use_cudnn=True):
		super(log_softmax, self).__init__()
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.log_softmax(x, self.use_cudnn)

class maxout(Function):
	def __init__(self, pool_size, axis=1):
		super(maxout, self).__init__()
		self.pool_size = pool_size
		self.axis = axis

	def __call__(self, x):
		return F.maxout(x, self.pool_size, self.axis)

class relu(Function):
	def __init__(self, use_cudnn=True):
		super(relu, self).__init__()
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.relu(x, self.use_cudnn)

class sigmoid(Function):
	def __init__(self, use_cudnn=True):
		super(sigmoid, self).__init__()
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.sigmoid(x, self.use_cudnn)

class softmax(Function):
	def __init__(self, use_cudnn=True):
		super(softmax, self).__init__()
		self.use_cudnn = use_cudnn
		pass
	def __call__(self, x):
		return F.softmax(x, self.use_cudnn)

class softplus(Function):
	def __init__(self, use_cudnn=True):
		super(softplus, self).__init__()
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.softplus(x, self.use_cudnn)

class tanh(Function):
	def __init__(self, use_cudnn=True):
		super(tanh, self).__init__()
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.tanh(x, self.use_cudnn)

class dropout(Function):
	def __init__(self, ratio=0.5):
		super(dropout, self).__init__()
		self.ratio = ratio

	def __call__(self, x, test=False):
		return F.dropout(x, self.ratio, test)

class average_pooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, use_cudnn=True):
		super(average_pooling_2d, self).__init__()
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.average_pooling_2d(x, self.ksize, self.stride, self.pad, self.use_cudnn)

class average_pooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, use_cudnn=True):
		super(average_pooling_2d, self).__init__()
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.average_pooling_2d(x, self.ksize, self.stride, self.pad, self.use_cudnn)

class max_pooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
		super(max_pooling_2d, self).__init__()
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.cover_all = cover_all
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.max_pooling_2d(x, self.ksize, self.stride, self.pad, self.cover_all, self.use_cudnn)

class spatial_pyramid_pooling_2d(Function):
	def __init__(self, pyramid_height, pooling_class, use_cudnn=True):
		super(spatial_pyramid_pooling_2d, self).__init__()
		self.pyramid_height = pyramid_height
		self.pooling_class = pooling_class
		self.use_cudnn = use_cudnn

	def __call__(self, x):
		return F.spatial_pyramid_pooling_2d(x, self.pyramid_height, self.pooling_class, self.use_cudnn)

class unpooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		super(unpooling_2d, self).__init__()
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return F.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class unpooling_2d(Function):
	def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
		super(unpooling_2d, self).__init__()
		self.ksize = ksize
		self.stride = stride
		self.pad = pad
		self.outsize = outsize
		self.cover_all = cover_all

	def __call__(self, x):
		return F.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)