import copy, json, types
import chainer
import link
import function

class Chain(chainer.Chain):

	def __init__(self, layers):
		super(Chain, self).__init__()
		self.n_layers = len(layers)
		for name, layer in layers.iteritems():
			if isinstance(layer, chainer.link.Link):
				self.add_link(name, layer)
			else:
				setattr(self, name, layer)

	def __call__(self, x, test=False):
		for i in xrange(self.n_layers):
			layer = getattr(self, "layer_%d" % i)
			if isinstance(layer, function.dropout):
				x = layer(x, train=not test)
			elif isinstance(layer, chainer.links.BatchNormalization):
				x = layer(x, test=test)
			else:
				x = layer(x)
		return x

class Sequential(object):
	def __init__(self, weight_initializer="Normal", weight_init_std=1):
		self._layers = []
		self.chain = None

		self.weight_initializer = weight_initializer	# Normal / GlorotNormal / HeNormal
		self.weight_init_std = weight_init_std

	def add(self, layer):
		if isinstance(layer, link.Link) or isinstance(layer, function.Function):
			self._layers.append(layer)
		elif isinstance(layer, function.Activation):
			self._layers.append(layer.to_function())
		else:
			raise Exception()

	def layer_from_dict(self, dict):
		if "_link" in dict:
			if hasattr(link, dict["_link"]):
				args = self.dict_to_layer_args(dict)
				return getattr(link, dict["_link"])(**args)
		if "_function" in dict:
			if hasattr(function, dict["_function"]):
				args = self.dict_to_layer_args(dict)
				return getattr(function, dict["_function"])(**args)
		raise Exception()

	def dict_to_layer_args(self, dict):
		args = copy.deepcopy(dict)
		remove_keys = []
		for key, value in args.iteritems():
			if key[0] == "_":
				remove_keys.append(key)
		for key in remove_keys:
			del args[key]
		return args

	def get_weight_initializer(self):
		if self.weight_initializer.lower() == "normal":
			return chainer.initializers.Normal(self.weight_init_std)
		if self.weight_initializer.lower() == "glorotnormal":
			return chainer.initializers.GlorotNormal(self.weight_init_std)
		if self.weight_initializer.lower() == "henormal":
			return chainer.initializers.HeNormal(self.weight_init_std)
		raise Exception()

	def layer_to_chainer_link(self, layer):
		if hasattr(layer, "_link"):
			if layer.has_multiple_weights() == True:
				if isinstance(layer, link.GRU):
					layer._init = self.get_weight_initializer()
					layer._inner_init = self.get_weight_initializer()
				elif isinstance(layer, link.LSTM):
					layer._lateral_init  = self.get_weight_initializer()
					layer._upward_init  = self.get_weight_initializer()
					layer._bias_init = self.get_weight_initializer()
					layer._forget_bias_init = self.get_weight_initializer()
				elif isinstance(layer, link.StatelessLSTM):
					layer._lateral_init  = self.get_weight_initializer()
					layer._upward_init  = self.get_weight_initializer()
				elif isinstance(layer, link.StatefulGRU):
					layer._init = self.get_weight_initializer()
					layer._inner_init = self.get_weight_initializer()
			else:
				layer._initialW = self.get_weight_initializer()
			return layer.to_link()
		if hasattr(layer, "_function"):
			return layer
		raise Exception()

	def build(self):
		json = self.to_json()
		self.from_json(json)

	def to_json(self):
		result = []
		for layer in self._layers:
			config = layer.to_dict()
			dict = {}
			for key, value in config.iteritems():
				if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
					dict[key] = value
			result.append(dict)

		return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

	def from_json(self, str):
		self.chain = None
		self._layers = []
		attributes = {}
		dict_array = json.loads(str)
		for i, dict in enumerate(dict_array):
			layer = self.layer_from_dict(dict)
			link = self.layer_to_chainer_link(layer)
			attributes["layer_%d" % i] = link
			self._layers.append(layer)
		self.chain = Chain(attributes)

	def __call__(self, x, test=False):
		return self.chain(x, test)
