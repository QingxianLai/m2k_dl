require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

local Provider = torch.class 'Provider'

local function _rand(max_val)
    return math.floor(torch.rand(1)[1] * max_val + 1)
end

function data_augmentation(src, width, height, max_expand, max_rotate, max_shift)
    -- expand
    local _expand = image.scale(_rand(width * max_expand))
    local x1 = _rand(_expand - width)
    local y1 = _rand(_expand - height)
    src = image.crop(src, x1, y1, x1+width, y1 + height) 

    -- shift
    src = image.translate(src, max_shift * _balance_rand(), max_shift * _balance_rand())
    
    -- rotate
    src = image.rotate(src, max_rotate * _balance_rand())

    -- contrast1
    

    -- contrast2
    return src
end

function Provider:__init(full)
    local exsize = 100000
    local channel = 3
    local height = 96
    local width = 96
    
    local psize = 400
    local K = 150 --augmented versions for each patch
    local pheight = 32
    local pwidth = 32

    if not paths.dirp('stl-10') then 
        print("fail to read data")
        return

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
        local this_d = raw_extra[_rand(exsize)]
        local x1 = _rand(width - pwidth)
        local y1 = _rand(height - pheight)
        local patch = image.crop(this_d, x1, y1, x1+pwidth, y1+pheight)
        t[idx]:copy(patch)
        l[idx] = idx
        

