require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')


local function _rand(max_val)
    return math.floor(torch.rand(1)[1] * max_val + 1)
end

local function _balance_rand()
    return torch.rand(1)[1] * 2 - 1
end

function _contrast(src, c)
    F = (259 * (c + 225)) /(255 * (259 - c))
    local new_image = torch.Tensor(src:size()):copy(src)
    for i = 1, src:size()[1] do
        local mean = src[i]:mean()
        new_image[i]:add(-mean):mul(F):add(mean)
        new_image[i]:clamp(-255,255)
    end
    return new_image
end


function data_augmentation(src, width, height, max_expand, max_rotate, max_shift, max_contrast)
    -- expand
    local exVal =  width * _rand(max_expand)
    local _expand = image.scale(src, exVal)
    local x1 = _rand(exVal - width-1)
    local y1 = _rand(exVal - height-1)
    src = image.crop(_expand, x1, y1, x1+width, y1 + height) 

    -- shift
    src = image.translate(src, max_shift * _balance_rand(), max_shift * _balance_rand())
    
    -- rotate
    src = image.rotate(src, max_rotate * _balance_rand())

    -- contrast
    src = _contrast(src, max_contrast*_balance_rand()) 

    return src
end


local Provider = torch.class 'Provider'

function Provider:__init(full)
    local exsize = 100000
    local channel = 3
    local height = 96
    local width = 96
    
    local psize = 4096
    local K = 128 --augmented versions for each patch
    local pheight = 32
    local pwidth = 32

    --if not paths.dirp('stl-10') then 
        --print("fail to read data")
        --return
    --end

    local raw_extra = torch.load('./stl-10/extra.t7b')

    -- load and parse the dataset
    self.unlabelData = {
        data = torch.Tensor(),
        labels = torch.Tensor(),
        size = function() return exsize end
    }

    local t = torch.ByteTensor(psize*K, channel, pheight, pwidth)
    local l = torch.ByteTensor(psize*K)
    local idx = 1
    for i = 1, psize do
        io.write("\n" .. idx)
        local this_d = raw_extra.data[1][_rand(exsize)]
        this_d = this_d:float()
        local x1 = _rand(width - pwidth)
        local y1 = _rand(height - pheight)
        local patch = image.crop(this_d, x1, y1, x1+pwidth, y1+pheight)
        t[idx]:copy(patch)
        l[idx] = idx
        for j = 1,K-1 do
            local new_patch = data_augmentation(patch,pwidth,pheight,1.3,0.2,3,150)
            t[j*psize + idx]:copy(new_patch)
            l[j*psize + idx] = idx
        end
        idx = idx + 1
        
    end
    self.unlabelData.data = t:float()
    self.unlabelData.labels = l:float()

    self.unlabelData.size = function() return psize * K end

end

function Provider:normalize()
    local data = self.unlabelData
    print '<trainer> preprocessing data (normalization)'
    collectgarbage()

    for i = 1,3 do
        local mean = data.data:select(2,i):mean()
        data.data:select(2,i):add(-mean)
    end
end
