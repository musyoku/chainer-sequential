def get_conv_outsize(in_size, ksize, stride, padding, cover_all=False, d=1):
	dk = ksize + (ksize - 1) * (d - 1)
	if cover_all:
		return (in_size + padding * 2 - dk + stride - 1) // stride + 1
	else:
		return (in_size + padding * 2 - dk) // stride + 1

def get_conv_padding(in_size, ksize, stride):
	pad2 = stride - (in_size - ksize) % stride
	if pad2 % stride == 0:
		return 0
	if pad2 % 2 == 1:
		return pad2
	return pad2 / 2

def get_deconv_padding(in_size, out_size, ksize, stride, cover_all=False):
	if cover_all:
		return (stride * (in_size - 1) + ksize - stride + 1 - out_size) // 2
	else:
		return (stride * (in_size - 1) + ksize - out_size) // 2

def get_deconv_outsize(in_size, ksize, stride, padding, cover_all=False):
	if cover_all:
		return stride * (in_size - 1) + ksize - stride + 1 - 2 * padding
	else:
		return stride * (in_size - 1) + ksize - 2 * padding

def get_deconv_insize(out_size, ksize, stride, padding, cover_all=False):
	if cover_all:
		return (out_size - ksize + stride - 1 + 2 * padding) // stride + 1
	else:
		return (out_size - ksize + 2 * padding) // stride + 1