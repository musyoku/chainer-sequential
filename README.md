Chainerのネットワーク構造をKerasのように書きたかったので作りました。

内部的にはChainerのLinkやFunctionをユーザーが定義したとおりの順に並べて実行します。

## Requirements
- Chainer 1.21

## Usage

```
from layers import Linear, BatchNormalization
from functions import Activation
from chain import Chain

x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

model = Sequential()
model.add(Linear(28*28, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(None, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(500, 28*28, use_weightnorm=True))
model.build(weight_initializer="GlorotNormal", weight_std=0.05)

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
model.build("Normal", 1.0)

for i, link in enumerate(model.links):
	if isinstance(link, chainer.link.Link):
		...
```

## JSON

```
json_str = model.to_json()
model.from_json(json_str)
```

## Adding activation function

```
model.add(functions.Activation("relu"))
```

or

```
model.add(functions.relu())
```

## Dropout

```
model.add(functions.dropout())
```

## Adding gaussian noise

```
model.add(functions.gaussian_noise(std=0.5))
```

## Weight Normalization

```
model.add(layers.Linear(500, 500, use_weightnorm=True))
model.add(layers.Convolution2D(64, 128, ksize=4, stride=2, pad=0, use_weightnorm=True))
model.add(layers.Deconvolution2D(64, 32, ksize=4, stride=2, pad=1, use_weightnorm=True))
```

## Initializing weights

```
model.build(weight_initializer="Normal", weight_std=1.0)
model.build(weight_initializer="GlorotNormal", weight_std=0.5)
model.build(weight_initializer="HeNormal", weight_std=0.1)
```

## Minibatch Discrimination

```
model.add(layers.Convolution2D(256, 1024, ksize=4, stride=2, pad=1))
model.add(functions.reshape_1d())
model.add(layers.MinibatchDiscrimination(None, num_kernels=50, ndim_kernel=5))
```

## Merge inputs

```
model.add(layers.Merge(num_inputs=2, out_size=500))
model.build("Normal", 1.0)
x = ...
y = ...
output = model(x, y)
```

## Residual connections

```
seq = Sequential()
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
res = Residual()
res.add(layers.Linear(500, 500))
res.add(layers.BatchNormalization(500))
res.add(functions.Activation("relu"))
seq.add(res)
seq.build("Normal", 1)

x = Variable(np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32))
y = seq(x)
```

y = res(seq(x)) + seq(x)

## PixelShuffler Layer

```
input_size = 2
seq = Sequential()
seq.add(layers.Linear(100, 64 * input_size ** 2))
seq.add(layers.BatchNormalization(64 * input_size ** 2))
seq.add(functions.Activation("relu"))
seq.add(functions.reshape((-1, 64, input_size, input_size)))
seq.add(layers.PixelShuffler2D(64, 32, r=2))
seq.add(layers.BatchNormalization(32))
seq.add(functions.Activation("relu"))
seq.add(layers.PixelShuffler2D(32, 16, r=2))
seq.add(layers.BatchNormalization(16))
seq.add(functions.Activation("relu"))
seq.add(layers.PixelShuffler2D(16, 3, r=2))
```

## DCGAN

```
import util

image_width = 96

disciminator = Sequential()
disciminator.add(Convolution2D(3, 64, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Convolution2D(64, 128, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Convolution2D(128, 256, ksize=4, stride=2, pad=1))
disciminator.add(Activation("elu"))
disciminator.add(Linear(None, 1))
disciminator.add(sigmoid())
disciminator.build("Normal", 0.1)

# compute projection width
projection_width = util.get_in_size_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

# compute required paddings
paddings = util.get_paddings_of_deconv_layers(image_width, num_layers=3, ksize=4, stride=2)

generator = Sequential()
generator.add(Linear(100, 64 * projection_width ** 2))
generator.add(BatchNormalization(64 * projection_width ** 2))
generator.add(Activation("relu"))
generator.add(reshape((-1, 64, projection_width, projection_width)))
generator.add(Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(BatchNormalization(32))
generator.add(Activation("relu"))
generator.add(Deconvolution2D(32, 16, ksize=4, stride=2, pad=paddings.pop(0)))
generator.add(BatchNormalization(16))
generator.add(Activation("relu"))
generator.add(Deconvolution2D(16, 3, ksize=4, stride=2, pad=paddings.pop(0)))
generator.build("Normal", 1.0)
```

## Using the same initializer for all sequences

Add all sequence to chain without calling `build`.

```
seq1 = Sequential()
seq1.add(layers.Linear(28*28, 500))

seq2 = Sequential()
seq2.add(layers.Linear(28*28, 500))

seq3 = Sequential()
seq3.add(layers.Linear(28*28, 500))

chain = Chain("Normal", 1.0)
chain.add_sequence(seq1, name="seq1")
chain.add_sequence(seq2, name="seq2")
chain.add_sequence(seq3, name="seq3")

# check
for link in chain.seq1.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq2.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq3.links:
	print np.std(link.W.data), np.mean(link.W.data)
```

To set the initial std of weights to different values,

```
seq1 = Sequential(weight_std=1.0)
seq1.add(layers.Linear(28*28, 500))

seq2 = Sequential(weight_std=0.1)
seq2.add(layers.Linear(28*28, 500))

seq3 = Sequential(weight_std=0.01)
seq3.add(layers.Linear(28*28, 500))

chain = Chain("Normal", 1.0)
chain.add_sequence(seq1, name="seq1")
chain.add_sequence(seq2, name="seq2")
chain.add_sequence(seq3, name="seq3")

# check
for link in chain.seq1.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq2.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq3.links:
	print np.std(link.W.data), np.mean(link.W.data)
```