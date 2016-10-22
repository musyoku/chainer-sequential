import unittest
import numpy as np
import chainer
from chainer import cuda, Variable, gradient_check, testing
from chainer.testing import attr, condition, parameterize
from chainer.utils import conv
import deconvolution_2d

def _pair(x):
	if hasattr(x, '__getitem__'):
		return x
	return x, x

@parameterize(
	*testing.product({
		'in_channels': [3],
		'out_channels': [2],
		'ksize': [3],
		'stride': [2],
		'pad': [1],
		'nobias': [True, False],
		'use_cudnn': [True, False]
	})
)
class TestDeconvolution2D(unittest.TestCase):

	def setUp(self):
		self.link = deconvolution_2d.Deconvolution2D(
			self.in_channels, self.out_channels, self.ksize,
			stride=self.stride, pad=self.pad, nobias=self.nobias)
		self.link.V.data[...] = np.random.uniform(-1, 1, self.link.V.data.shape).astype(np.float32)


		N = 2
		h, w = 3, 2
		kh, kw = _pair(self.ksize)
		out_h = conv.get_deconv_outsize(h, kh, self.stride, self.pad)
		out_w = conv.get_deconv_outsize(w, kw, self.stride, self.pad)
		self.gy = np.random.uniform(-1, 1, (N, self.out_channels, out_h, out_w)).astype(np.float32)
		self.x = np.random.uniform(-1, 1, (N, self.in_channels, h, w)).astype(np.float32)

		self.link(Variable(self.x))
		if not self.nobias:
			self.link.b.data[...] = np.random.uniform(-1, 1, self.link.b.data.shape).astype(np.float32)
		self.link.cleargrads()
		self.check_backward_options = {"atol": 1e-5, "rtol": 3e-4}

	def check_forward_consistency(self):
		x_cpu = Variable(self.x)
		y_cpu = self.link(x_cpu)
		self.assertEqual(y_cpu.data.dtype, np.float32)

		self.link.to_gpu()
		x_gpu = Variable(cuda.to_gpu(self.x))
		y_gpu = self.link(x_gpu)
		self.assertEqual(y_gpu.data.dtype, np.float32)

		testing.assert_allclose(y_cpu.data, y_gpu.data.get())

	@attr.gpu
	@condition.retry(3)
	def test_forward_consistency(self):
		self.link.use_cudnn = self.use_cudnn
		self.check_forward_consistency()

	def check_backward(self, x_data, y_grad):
		params = [self.link.V, self.link.g]
		if not self.nobias:
			params.append(self.link.b)

		gradient_check.check_backward(self.link, x_data, y_grad, params, eps=1e-2, atol=1e-4, rtol=1e-4)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.use_cudnn = self.use_cudnn
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)