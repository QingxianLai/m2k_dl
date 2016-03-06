require 'nn'
require 'image'
require 'xlua'
require 'unsup'

opt = lapp[[
    -d,--data      (default "./stl-10/extra.t7b")     input data
]]

torch.setdefaulttensortype('torch.FloatTensor')

local Provider = torch.class 'Provider'

function Provider:__init(full)
    local extraSize = 100000
    local channel = 3
    local height = 96
    local width = 96

    local K = 10
    
    local raw_extra = torch.load(opt.data)

    self.extraData = {
        data = torch.Tensor(),
    }

    local t = torch.FloatTensor(#raw_extra.data[1], channel*width*height)
    for i = 1,#raw_extra.data[1] do 
        t[i]:copy(raw_extra.data[1][i]:reshape(1, channel*width*height))
    end

    self.extraData.data = t:float()
end

    
