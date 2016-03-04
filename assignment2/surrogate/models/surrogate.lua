require 'nn'

local sur = nn.Sequential()

local MaxPooling = nn.SpatialMaxPooling


sur:add(nn.SpatialConvolution(3,64, 5,5, 1,1, 2,2))
sur:add(nn.SpatialBatchNormalization(64, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.3))
sur:add(MaxPooling(2,2,2,2):ceil())

sur:add(nn.SpatialConvolution(64,128, 3,3, 1,1, 1,1))
sur:add(nn.SpatialBatchNormalization(128, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.4))
sur:add(MaxPooling(2,2,2,2):ceil())

sur:add(nn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1))
sur:add(nn.SpatialBatchNormalization(256, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.4))
sur:add(MaxPooling(2,2,2,2):ceil())

sur:add(nn.SpatialConvolution(256, 512, 3,3, 1,1, 1,1))
sur:add(nn.SpatialBatchNormalization(512, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.4))
sur:add(MaxPooling(2,2,2,2):ceil())

sur:add(nn.View(512*2*2))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*2*2,512*2*2))
classifier:add(nn.BatchNormalization(512*2*2))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*2*2,4096))

sur:add(classifier)


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(sur)

-- check that we can propagate forward without errors
-- should get 16x4096 tensor
require 'cutorch'
require 'cunn'
print(#sur:cuda():forward(torch.CudaTensor(16,3,32,32)))

return sur
