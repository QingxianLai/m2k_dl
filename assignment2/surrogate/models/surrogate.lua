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
sur:add(MaxPooling(2,2,2,2))

sur:add(nn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1))
sur:add(nn.SpatialBatchNormalization(256, 1e-3))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.4))


