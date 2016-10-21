import copy
import json
import link
import function

class Sequential(object):
	def __init__(self):
		self._layers = []
		self.sequence = []

	def add(self, layer):
		if isinstance(layer, link.Link) or isinstance(layer, function.Function):
			self._layers.append(layer)
		else:
			raise Exception()

	def layer_from_dict(self, dict):
		if dict["_link"] is not None:
			if hasattr(link, dict["_link"]):
				args = self.dict_to_layer_args(dict)
				return getattr(link, dict["_link"])(**args)
		if dict["_function"] is not None:
			if hasattr(function, dict["_function"]):
				args = self.dict_to_layer_args(dict)
				return getattr(function, dict["_function"])(**args)
		raise Exception()

	def dict_to_layer_args(self, dict):
		args = copy.deepcopy(dict)
		if "_link" in args:
			del args["_link"]
		if "_function" in args:
			del args["_function"]
		return args

	def layer_to_chainer_link(self, layer):
		if hasattr(layer, "_link"):
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
			result.append(layer.to_dict())

		return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

	def from_json(self, str):
		self.sequence = []
		self._layers = []
		a = json.loads(str)
		for dict in a:
			layer = self.layer_from_dict(dict)
			link = self.layer_to_chainer_link(layer)
			self.sequence.append(link)
			self._layers.append(layer)

	def __call__(self, x):
		for layer in self.sequence:
			x = layer(x)
		return x
