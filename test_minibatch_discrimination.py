import os
import tempfile
import unittest
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import gradient_check
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check
import minibatch_discrimination

@testing.parameterize(*testing.product({
	"in_shape": [(3,), (3, 2, 2)],
	"x_dtype": [np.float16, np.float32, np.float64],
	"W_dtype": [np.float16, np.float32, np.float64],
}))
class TestLinear(unittest.TestCase):

	num_kernels = 2
	ndim_kernel = 3

	def setUp(self):
		in_size = np.prod(self.in_shape)
		out_size = in_size + self.num_kernels
		self.link = minibatch_discrimination.MinibatchDiscrimination(in_size, self.num_kernels, self.ndim_kernel, initialV=chainer.initializers.Normal(1, self.W_dtype))

		x_shape = (4,) + self.in_shape
		self.x = np.random.uniform(-1, 1, x_shape).astype(self.x_dtype)

		self.link(Variable(self.x))

		T = self.link.T.data
		self.link.cleargrads()

		self.gy = np.random.uniform(-1, 1, (4, out_size)).astype(self.x_dtype)

		batchsize = self.x.shape[0]
		kernels = np.expand_dims(np.tensordot(self.x, T, axes=(1, 0)), 3)
		kernels_t = np.transpose(kernels, (3, 1, 2, 0))
		kernels, kernels_t = np.broadcast_arrays(kernels, kernels_t)
		c_b = np.sum(abs(kernels - kernels_t), axis=2)
		eraser = np.broadcast_to(np.eye(batchsize).reshape((batchsize, 1, batchsize)), c_b.shape)
		self.y = np.append(self.x, np.sum(np.exp(-(c_b + 1e6 * eraser)), axis=2), axis=1)

		self.check_forward_options = {}
		self.check_backward_options = {}
		if self.x_dtype == np.float16:
			self.check_forward_options = {"atol": 1e-3, "rtol": 1e-2}
			self.check_backward_options = {"atol": 1e-2, "rtol": 5e-2}
		elif self.W_dtype == np.float16:
			self.check_backward_options = {"atol": 1e-2, "rtol": 5e-2}

	def check_forward(self, x_data):
		x = Variable(x_data)
		y = self.link(x)
		self.assertEqual(y.data.dtype, self.x_dtype)
		testing.assert_allclose(self.y, y.data, **self.check_forward_options)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.x)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.link.to_gpu()
		self.check_forward(cuda.to_gpu(self.x))

	def check_backward(self, x_data, y_grad):
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.T,), eps=2 ** -3, **self.check_backward_options)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

class TestLinearParameterShapePlaceholder(unittest.TestCase):

	in_size = 3
	num_kernels = 2
	ndim_kernel = 3
	in_shape = (in_size,)
	out_size = in_size + num_kernels
	in_size_or_none = None

	def setUp(self):
		self.link = minibatch_discrimination.MinibatchDiscrimination(self.in_size_or_none, self.num_kernels, self.ndim_kernel)
		temp_x = np.random.uniform(-1, 1, (self.out_size, self.in_size)).astype(np.float32)

		self.link(Variable(temp_x))
		T = self.link.T.data
		T[...] = np.random.uniform(-1, 1, T.shape).astype(np.float32)
		self.link.cleargrads()

		x_shape = (4,) + self.in_shape
		self.x = np.random.uniform(-1, 1, x_shape).astype(np.float32)
		self.gy = np.random.uniform(-1, 1, (4, self.out_size)).astype(np.float32)

		batchsize = self.x.shape[0]
		kernels = np.expand_dims(np.tensordot(self.x, T, axes=(1, 0)), 3)
		kernels_t = np.transpose(kernels, (3, 1, 2, 0))
		kernels, kernels_t = np.broadcast_arrays(kernels, kernels_t)
		c_b = np.sum(abs(kernels - kernels_t), axis=2)
		eraser = np.broadcast_to(np.eye(batchsize).reshape((batchsize, 1, batchsize)), c_b.shape)
		self.y = np.append(self.x, np.sum(np.exp(-(c_b + 1e6 * eraser)), axis=2), axis=1)

	def check_forward(self, x_data):
		x = Variable(x_data)
		y = self.link(x)
		self.assertEqual(y.data.dtype, np.float32)
		testing.assert_allclose(self.y, y.data)

	@condition.retry(3)
	def test_forward_cpu(self):
		self.check_forward(self.x)

	@attr.gpu
	@condition.retry(3)
	def test_forward_gpu(self):
		self.link.to_gpu()
		self.check_forward(cuda.to_gpu(self.x))

	def check_backward(self, x_data, y_grad):
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.T,), eps=1e-2, atol=1e-4, rtol=1e-3)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	def test_serialization(self):
		lin1 = minibatch_discrimination.MinibatchDiscrimination(None, self.num_kernels, self.ndim_kernel)
		x = Variable(self.x)
		# Must call the link to initialize weights.
		lin1(x)
		w1 = lin1.T.data
		fd, temp_file_path = tempfile.mkstemp()
		os.close(fd)
		npz.save_npz(temp_file_path, lin1)
		lin2 = minibatch_discrimination.MinibatchDiscrimination(None, self.num_kernels, self.ndim_kernel)
		npz.load_npz(temp_file_path, lin2)
		w2 = lin2.T.data
		self.assertEqual((w1 == w2).all(), True)

class TestInvalidLinear(unittest.TestCase):

	def setUp(self):
		self.link = minibatch_discrimination.MinibatchDiscrimination(3, 2)
		self.x = np.random.uniform(-1, 1, (4, 1, 2)).astype(np.float32)

	def test_invalid_size(self):

		with self.assertRaises(type_check.InvalidType):
			self.link(Variable(self.x))

testing.run_module(__name__, __file__)