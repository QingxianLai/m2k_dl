require 'cunn'
require 'cutorch'
require 'image'

local model = torch.load("./default_add_data/logs/model.net")

local w = model:get(1).weight
print(w:size())
dis = image.toDisplayTensor{input=w, padding=1, nrow=8}
image.save("./filters/baseline_add_data.png", dis)
