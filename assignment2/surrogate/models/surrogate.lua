require 'nn'

local sur = nn.Sequential()

local function ConvBNReLU(nInputPlane, nOutputPlane, )
    sur.add(nn.SpatialConvolution(nInputPlane, nOutputPlane, ))

local MaxPooling = nn.SpatialMaxPooling

sur:add(nn.SpatialConvolution(3,64, 3,3, 1,1, 1,1))
sur:add(nn.SpatialBatchNormalization(64, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.3))



