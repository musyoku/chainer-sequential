import math
import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F
from chainer import cuda, Variable
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import array
from chainer.utils import type_check

class MinibatchDiscrimination(chainer.chain):
	def __init__(self, in_size, num_kernels, ndim_kernel=5, wscale=1, initialT=None):
		super(MinibatchDiscrimination, self).__init__(
			T=L.linear(in_size, num_kernels * ndim_kernel, wscale=wscale, initialW=initialT)
		)

		self.num_kernels = num_kernels
		self.ndim_kernel = ndim_kernel

	def __call__(self, x):
		xp = cuda.get_array_module(x.data)
		M = F.reshape(self.T(x), (-1, self.num_kernels, self.ndim_kernel))
		M = F.expand_dims(M, 3)
		M_T = F.transpose(M, (3, 1, 2, 0))
		M, M_T = F.broadcast(M, M_T)

		norm = F.sum(abs(M - M_T), axis=2)
		eraser = F.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), norm.shape)
		c_b = F.exp(-(norm + 1e6 * eraser))
		o_b = F.sum(c_b, axis=2)
		return F.concat((x, o_b), axis=1)

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

def get_norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=1)) + 1e-9
	norm = norm.reshape((-1, 1))
	return norm

class MinibatchDiscriminationFunction(function.Function):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		x_type, T_type = in_types

		type_check.expect(
			x_type.dtype.kind == "f",
			T_type.dtype.kind == "f",
			x_type.ndim >= 2,
			T_type.ndim == 3,
			type_check.prod(x_type.shape[1:]) == T_type.shape[0],
		)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		T = inputs[1]
		xp = cuda.get_array_module(T)
		batchsize = x.shape[0]

		# print "[T]"
		# print T
		# print T.shape

		M = xp.tensordot(x, T, axes=(1, 0))

		# A slower but equivalent way of computing the same...
		# _kernels = xp.zeros((x.shape[0], T.shape[1], T.shape[2]))
		# for i in xrange(x.shape[0]):
		# 	for j in xrange(T.shape[2]):
		# 		for k in xrange(T.shape[1]):
		# 			for l in xrange(T.shape[0]):
		# 				_kernels[i, k, j] += x[i, l] * T[l, k, j]

		M = xp.expand_dims(M, 3)
		M_T = xp.transpose(M, (3, 1, 2, 0))
		M, M_T = xp.broadcast_arrays(M, M_T)

		# print M.shape
		# print M_T
		# print M_T.shape

		self.diff = M - M_T
		norm = xp.sum(abs(M - M_T), axis=2)
		self.norm = norm
		# print norm
		eraser = xp.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), norm.shape)
		# print eraser
		c_b = xp.exp(-(norm + 1e6 * eraser))
		# print c_b
		o_b = xp.sum(c_b, axis=2)
		# print o_b
		# print x
		y = xp.append(x, o_b, axis=1)
		# print y

		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		T = inputs[1]
		xp = cuda.get_array_module(T)
		batchsize = x.shape[0]
		ndim_x = x.shape[1]
		num_kernels = T.shape[1]
		ndim_kernel = T.shape[2]

		gy = grad_outputs[0]

		print "\n[backward]"
		print x
		X = xp.expand_dims(x, 2)
		X_T = xp.transpose(X, (2, 1, 0))
		X, X_T = xp.broadcast_arrays(X, X_T)
		x_diff = X - X_T
		print x_diff
		print x_diff.shape
		print self.norm
		print self.norm.shape
		c_b = xp.exp(-self.norm)
		_c_b = c_b
		eraser = 1 - xp.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), self.norm.shape)
		c_b *= eraser
		print c_b

		print T.shape
		print c_b.shape
		print self.diff.shape
		print x_diff.shape
		print gy.shape

		gx = xp.zeros(x.shape, dtype=x.dtype)
		for m in xrange(batchsize):
			_x = x[m, :]
			for n in xrange(batchsize):
				if m == n:
					for o in xrange(batchsize):
						if m != o:
							for i in xrange(num_kernels):
								_gy = gy[n, ndim_x + i]
								_c_b_i = _c_b[n, i, o]
								for j in xrange(ndim_kernel):
									_norm_i = self.diff[n, i, j, o]
									for k in xrange(ndim_x):
										gx[m, k] += _gy * _c_b_i * xp.sign(_norm_i) * T[k, i, j]
				else:
					for i in xrange(num_kernels):
						_gy = gy[n, ndim_x + i]
						_c_b_i = _c_b[n, i, o]
						for j in xrange(ndim_kernel):
							_norm_i = self.diff[n, i, j, o]
							for k in xrange(ndim_x):
								gx[m, k] += _gy * _c_b_i * xp.sign(_norm_i) * T[k, i, j]
					pass
			
		gx[:] += gy[:, :ndim_x]

		gT = xp.zeros(T.shape, dtype=T.dtype)
		for i in xrange(gT.shape[0]):
			for j in xrange(gT.shape[1]):
				for k in xrange(gT.shape[2]):
					for a in xrange(batchsize):
						for b in xrange(batchsize):
							gT[i, j, k] += c_b[a, j, b] * x_diff[a, k, b]

		return gx, np.zeros_like(gT)

def minibatch_discrimination(x, T):
	return MinibatchDiscriminationFunction()(x, T)

class MinibatchDiscrimination(link.Link):

	def __init__(self, in_size, num_kernels, ndim_kernel=5, wscale=1, initialT=None):
		super(MinibatchDiscrimination, self).__init__()

		self.initialT = initialT
		self.wscale = wscale
		self.num_kernels = num_kernels
		self.ndim_kernel = ndim_kernel

		if in_size is None:
			self.add_uninitialized_param("T")
		else:
			self._initialize_weight(in_size)

	def _initialize_weight(self, in_size):
		self.add_param("T", (in_size, self.num_kernels, self.ndim_kernel), initializer=initializers._get_initializer(self.initialT, math.sqrt(self.wscale)))

	def __call__(self, x):
		if hasattr(self, "T") == False:
			with cuda.get_device(self._device_id):
				self._initialize_weight(np.prod(x.shape[1:]))

		return minibatch_discrimination(x, self.T)