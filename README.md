ChainerをKerasっぽく書きたいという思いから作りました。

ただしKerasを使ったことがないのであくまでそれっぽいというだけです。

## Usage

```
from link import Linear, BatchNormalization
from function import Activation
from chain import Chain

x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

model = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
model.add(Linear(28*28, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(None, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(500, 28*28, use_weightnorm=True))
model.build()

chain = Chain()
chain.add_sequence(model)
chain.setup_optimizers("adam", 0.001, momentum=0.9, weight_decay=0.000001, gradient_clipping=10)

for i in xrange(100):
	y = chain(x)
	loss = F.mean_squared_error(x, y)
	chain.backprop(loss)
	print float(loss.data)

chain.save("model")
```

## Getting chainer.links.Link objects

```
model = Sequential()
...
model.build()

for i, link in enumerate(model.links):
	if isinstance(link, chainer.link.Link):
		...
```

## Adding activation function

```
model.add(function.Activation("relu"))
```

or

```
model.add(function.relu())
```

## Dropout

```
model.add(function.dropout())
```

## Adding gaussian noise

```
model.add(function.gaussian_noise(std=0.5))
```

## Weight Normalization

```
model.add(link.Linear(500, 500, use_weightnorm=True))
model.add(link.Convolution2D(64, 128, ksize=4, stride=2, pad=0, use_weightnorm=True))
model.add(link.Deconvolution2D(64, 32, ksize=4, stride=2, pad=1, use_weightnorm=True))
```

## Initializing weights

```
model = Sequential(weight_initializer="Normal", weight_init_std=0.05)
model = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
model = Sequential(weight_initializer="HeNormal", weight_init_std=0.05)
```

## DCGAN

```
import util, link, function

image_width = 96

disciminator = Sequential()
disciminator.add(link.Convolution2D(3, 64, ksize=4, stride=2, pad=0))
disciminator.add(function.Activation("elu"))
disciminator.add(link.Convolution2D(64, 128, ksize=4, stride=2, pad=0))
disciminator.add(function.Activation("elu"))
disciminator.add(link.Convolution2D(128, 256, ksize=4, stride=2, pad=0))
disciminator.add(function.Activation("elu"))
disciminator.add(link.Linear(None, 1, use_weightnorm=True))
disciminator.add(function.sigmoid())
disciminator.build()

# compute projection width
input_width = util.get_in_size_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

# compute required paddings
paddings = util.get_paddings_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

generator = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
generator.add(link.Linear(100, 64 * input_width ** 2))
generator.add(link.BatchNormalization(64 * input_width ** 2))
generator.add(function.Activation("relu"))
generator.add(function.reshape((-1, 64, input_width, input_width)))
generator.add(link.Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(link.BatchNormalization(32))
generator.add(function.Activation("relu"))
generator.add(link.Deconvolution2D(32, 16, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(link.BatchNormalization(16))
generator.add(function.Activation("relu"))
generator.add(link.Deconvolution2D(16, 3, ksize=4, stride=2, pad=paddings.pop(0)))
generator.build()
```