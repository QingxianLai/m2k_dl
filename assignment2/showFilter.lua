require 'cunn'
require 'cutorch'
require 'image'

local model = torch.load("./default/logs/vgg_da/model.net")

local w = model:get(1).weight
print(w:size())
dis = image.toDisplayTensor{input=w, padding=1, nrow=8}
image.save("./filters/augmentation.png", dis)
