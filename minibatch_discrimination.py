import math
import numpy as np
from chainer import cuda, Variable
from chainer import initializers
from chainer import link
from chainer import function
from chainer.utils import array
from chainer.utils import type_check

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
		)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		T = inputs[1]
		xp = cuda.get_array_module(T)

		kernels = xp.tensordot(x, T, axes=(1, 0))
		print x.shape
		print T.shape
		print kernels.shape
		print kernels

		# A slower but equivalent way of computing the same...
		# _kernels = xp.zeros((x.shape[0], T.shape[1], T.shape[2]))
		# for i in xrange(x.shape[0]):
		# 	for j in xrange(T.shape[2]):
		# 		for k in xrange(T.shape[1]):
		# 			for l in xrange(T.shape[0]):
		# 				_kernels[i, k, j] += x[i, l] * T[l, k, j]
		# print _kernels


		self.normV = get_norm(T)
		self.normalizedV = T / self.normV
		self.W = g * self.normalizedV

		y = x.dot(self.W.T).astype(x.dtype, copy=False)
		if len(inputs) == 3:
			b = inputs[2]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		T = inputs[1]
		g = inputs[2]
		W = self.W
		xp = cuda.get_array_module(x)

		gy = grad_outputs[0]
		gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
		gW = gy.T.dot(x).astype(W.dtype, copy=False)

		gg = xp.sum(gW * self.normalizedV, axis=1, keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.normalizedV) / self.normV
		gV = gV.astype(T.dtype, copy=False)

		if len(inputs) == 4:
			gb = gy.sum(0)
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

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