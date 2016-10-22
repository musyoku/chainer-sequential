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
import linear

@testing.parameterize(*testing.product({
	"in_shape": [(3,), (3, 2, 2)],
	"x_dtype": [np.float16, np.float32, np.float64],
	"W_dtype": [np.float16, np.float32, np.float64],
}))
class TestLinear(unittest.TestCase):

	out_size = 2

	def setUp(self):
		in_size = np.prod(self.in_shape)
		self.link = linear.Linear(in_size, self.out_size, initialV=chainer.initializers.Normal(1, self.W_dtype), dtype=self.x_dtype)

		x_shape = (4,) + self.in_shape
		self.x = np.random.uniform(-1, 1, x_shape).astype(self.x_dtype)

		self.link(Variable(self.x))

		W = self.link._get_W_data()
		b = self.link.b.data
		self.link.cleargrads()

		self.gy = np.random.uniform(-1, 1, (4, self.out_size)).astype(self.x_dtype)
		self.y = self.x.reshape(4, -1).dot(W.T) + b
		self.check_forward_options = {}
		self.check_backward_options = {"atol": 1e-4, "rtol": 3e-3}
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
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.V, self.link.g, self.link.b), eps=2 ** -3, **self.check_backward_options)

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
	in_shape = (in_size,)
	out_size = 2
	in_size_or_none = None

	def setUp(self):
		self.link = linear.Linear(self.in_size_or_none, self.out_size)
		temp_x = np.random.uniform(-1, 1, (self.out_size, self.in_size)).astype(np.float32)

		self.link(Variable(temp_x))
		V = self.link.V.data
		V[...] = np.random.uniform(-1, 1, V.shape).astype(np.float32)
		g = self.link.g.data
		g[...] = np.random.uniform(-1, 1, g.shape).astype(np.float32)
		b = self.link.b.data
		b[...] = np.random.uniform(-1, 1, b.shape).astype(np.float32)

		W = g * V / np.linalg.norm(V)
		self.link.cleargrads()

		x_shape = (4,) + self.in_shape
		self.x = np.random.uniform(-1, 1, x_shape).astype(np.float32)
		self.x = np.ones(x_shape).astype(np.float32)
		self.gy = np.random.uniform(-1, 1, (4, self.out_size)).astype(np.float32)
		self.gy = np.ones((4, self.out_size)).astype(np.float32)
		self.y = self.x.reshape(4, -1).dot(W.T) + b

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
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.V, self.link.g, self.link.b), eps=1e-2, atol=1e-4, rtol=1e-3)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	def test_serialization(self):
		lin1 = linear.Linear(None, self.out_size)
		x = Variable(self.x)
		# Must call the link to initialize weights.
		lin1(x)
		w1 = lin1._get_W_data()
		fd, temp_file_path = tempfile.mkstemp()
		os.close(fd)
		npz.save_npz(temp_file_path, lin1)
		lin2 = linear.Linear(None, self.out_size)
		npz.load_npz(temp_file_path, lin2)
		w2 = lin2._get_W_data()
		self.assertEqual((w1 == w2).all(), True)

class TestInvalidLinear(unittest.TestCase):

	def setUp(self):
		self.link = linear.Linear(3, 2)
		self.x = np.random.uniform(-1, 1, (4, 1, 2)).astype(np.float32)

	def test_invalid_size(self):

		with self.assertRaises(type_check.InvalidType):
			self.link(Variable(self.x))

testing.run_module(__name__, __file__)