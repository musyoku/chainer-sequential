from sequential import Sequential
import link
import function

model = Sequential()
model.add(link.Bias())
model.add(link.Bilinear(1, 1, 1))
json = model.to_json()
print json
model.from_json(json)
json = model.to_json()
print json
model.from_json(json)