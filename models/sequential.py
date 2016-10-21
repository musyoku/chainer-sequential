class Sequential(Object):
	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.append(layer)

	def to_json(self):
		pass

	def from_json(self, json):
		pass