require 'nn'

local sur = nn.Sequential()

local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))

sur:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,128)

sur:add(MaxPooling(2,2,2,2):ceil())

sur:add(nn.Dropout(0.4))
sur:add(nn.Linear(128, 256))
sur:add(nn.BatchNormalization(128))
sur:add(nn.ReLU(true))
sur:add(nn.Dropout(0.5))
sur:add(nn.Linear())



