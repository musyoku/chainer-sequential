import numpy as np
import six.moves.cPickle as pickle
import unittest
import chainer
from chainer import cuda, Variable, testing, gradient_check
from chainer.testing import attr, condition
from chainer.utils import conv
import convolution_2d

@testing.parameterize(*testing.product({
	"x_dtype": [np.float16, np.float32, np.float64],
	"W_dtype": [np.float16, np.float32, np.float64],
}))
class TestConvolution2D(unittest.TestCase):

	def setUp(self):
		self.link = convolution_2d.Convolution2D(
			3, 2, 3, stride=2, pad=1,
			initialV=chainer.initializers.Normal(1, self.W_dtype),
			dtype=self.x_dtype)

		self.x = np.random.uniform(-1, 1, (2, 3, 4, 3)).astype(self.x_dtype)
		self.link(Variable(self.x))
		self.link.cleargrads()
		self.gy = np.random.uniform(-1, 1, (2, 2, 2, 2)).astype(self.x_dtype)
		self.check_forward_options = {}
		self.check_backward_options = {"atol": 1e-5, "rtol": 3e-4}
		if self.x_dtype == np.float16 or self.W_dtype == np.float16:
			self.check_backward_options = {"atol": 3e-2, "rtol": 5e-2}

	@attr.gpu
	def test_im2col_consistency(self):
		col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
		col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 1, 1)
		testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

	@attr.gpu
	def test_col2im_consistency(self):
		col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
		h, w = self.x.shape[2:]
		im_cpu = conv.col2im_cpu(col, 2, 2, 1, 1, h, w)
		im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 1, 1, h, w)
		testing.assert_allclose(im_cpu, im_gpu.get())

	def check_forward_consistency(self):
		x_cpu = Variable(self.x)
		y_cpu = self.link(x_cpu)
		self.assertEqual(y_cpu.data.dtype, self.x_dtype)

		self.link.to_gpu()
		x_gpu = Variable(cuda.to_gpu(self.x))
		y_gpu = self.link(x_gpu)
		self.assertEqual(y_gpu.data.dtype, self.x_dtype)

		testing.assert_allclose(y_cpu.data, y_gpu.data.get())

	@attr.gpu
	@condition.retry(3)
	def test_forward_consistency(self):
		self.check_forward_consistency()

	@attr.gpu
	@condition.retry(3)
	def test_forward_consistency_im2col(self):
		self.link.use_cudnn = False
		self.check_forward_consistency()

	def check_backward(self, x_data, y_grad):
		gradient_check.check_backward(
			self.link, x_data, y_grad, (self.link.V, self.link.g, self.link.b), eps=2 ** -3, **self.check_backward_options)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu_im2col(self):
		self.link.use_cudnn = False
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	def check_pickling(self, x_data):
		x = Variable(x_data)
		y = self.link(x)
		y_data1 = y.data

		del x, y

		pickled = pickle.dumps(self.link, -1)
		del self.link
		self.link = pickle.loads(pickled)

		x = Variable(x_data)
		y = self.link(x)
		y_data2 = y.data

		testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

	def test_pickling_cpu(self):
		self.check_pickling(self.x)

	@attr.gpu
	def test_pickling_gpu(self):
		self.link.to_gpu()
		self.check_pickling(cuda.to_gpu(self.x))


class TestConvolution2DParameterShapePlaceholder(unittest.TestCase):

	def setUp(self):
		in_channels = None
		self.link = convolution_2d.Convolution2D(in_channels, 2, 3, stride=2, pad=1)
		self.x = np.random.uniform(-1, 1, (2, 3, 4, 3)).astype(np.float32)
		self.link(Variable(self.x))
		b = self.link.b.data
		b[...] = np.random.uniform(-1, 1, b.shape)
		self.link.cleargrads()
		self.gy = np.random.uniform(-1, 1, (2, 2, 2, 2)).astype(np.float32)

	@attr.gpu
	def test_im2col_consistency(self):
		col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
		col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 1, 1)
		testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

	@attr.gpu
	def test_col2im_consistency(self):
		col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
		h, w = self.x.shape[2:]
		im_cpu = conv.col2im_cpu(col, 2, 2, 1, 1, h, w)
		im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 1, 1, h, w)
		testing.assert_allclose(im_cpu, im_gpu.get())

	def check_forward_consistency(self):
		x_cpu = Variable(self.x)
		y_cpu = self.link(x_cpu)
		self.assertEqual(y_cpu.data.dtype, np.float32)

		self.link.to_gpu()
		x_gpu = Variable(cuda.to_gpu(self.x))
		y_gpu = self.link(x_gpu)
		self.assertEqual(y_gpu.data.dtype, np.float32)

		testing.assert_allclose(y_cpu.data, y_gpu.data.get())

	@attr.cudnn
	@condition.retry(3)
	def test_forward_consistency(self):
		self.check_forward_consistency()

	@attr.gpu
	@condition.retry(3)
	def test_forward_consistency_im2col(self):
		self.link.use_cudnn = False
		self.check_forward_consistency()

	def check_backward(self, x_data, y_grad):
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.V, self.link.g, self.link.b), eps=1e-2, atol=3e-5, rtol=3e-4)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.cudnn
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu_im2col(self):
		self.link.use_cudnn = False
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	def check_pickling(self, x_data):
		x = Variable(x_data)
		y = self.link(x)
		y_data1 = y.data

		del x, y

		pickled = pickle.dumps(self.link, -1)
		del self.link
		self.link = pickle.loads(pickled)

		x = Variable(x_data)
		y = self.link(x)
		y_data2 = y.data

		testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

	def test_pickling_cpu(self):
		self.check_pickling(self.x)

	@attr.gpu
	def test_pickling_gpu(self):
		self.link.to_gpu()
		self.check_pickling(cuda.to_gpu(self.x))

testing.run_module(__name__, __file__)