require 'torch'
require 'image'
m = require 'manifold'



local test = torch.load("./stl-10/test.t7b")
local p = 0.125
data = torch.FloatTensor(8000 * p, 3, 96, 96)
idx = 1
for i = 1,10 do
    for j = 1,800*p do
        data[idx]:copy(test.data[i][j])
        idx = idx + 1
    end
end

data:resize(data:size()[1], data:size()[2] * data:size()[3] * data:size()[4])
print(data:size())

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(data, opts)

print(mapped_x1:size())

local im_size = 2048
map_im = m.draw_image_map(mapped_x1, data:resize(data:size(1), 3, 96, 96), im_size, 0, true)
collectgarbage()

image.save("./tsne_plot/test_origan.png", map_im)
