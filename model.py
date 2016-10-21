import json
import links
import functions

class Model(Object):
	def __init__(self):
		self._layers = []
		self.sequence = []

	def add(self, layer):
		if isinstance(layer, links.Link) or isinstance(layer, functions.Function):
			self._layers.append(layer)
		else:
			raise Exception()

	def build(self):
		json = self.to_json()
		self.from_json(json)

	def to_json(self):
		pass

	def from_json(self, json):
		pass