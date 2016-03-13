require 'torch'
require 'image'
m = require 'manifold'


-- load pictures 
local raw_data = torch.load("./stl-10/val.t7b")

local channel = 3
local width = 96
local height = 96

if #raw_data.data == 10 then
    -- train, val, test 
    local size = #raw_data.data[1] * 10
    pic = torch.ByteTensor(size, channel, width, height)
    
    local idx = 1
    for i = 1,10 do
        for j = 1,#raw_data.data[i] do
            pic[idx]:copy(raw_data.data[i][j])
            idx = idx + 1
        end
    end

else
    -- extra
    local size = #raw_data.data[1]
    pic = torch.ByteTensor(size, channel, width, height)
    for i=1,size do
        pic[i]:copy(raw_data.data[1][i])
    end
end
print(pic:size())


-- load lastLayer outputs
local data = torch.load("./last_layer.t7")
print(data:size())

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(data, opts)

print(mapped_x1:size())

local im_size = 4096
map_im = m.draw_image_map(mapped_x1, pic, im_size, 0, true)
collectgarbage()

image.save("./tsne_plot/val.png", map_im)
