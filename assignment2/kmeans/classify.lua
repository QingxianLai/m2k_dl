require 'xlua'
require 'nn'
require 'image'
require 'torch'
local c = require 'treql.colorize'

opt = lapp[[
    -l,--labels (default "./labels.t7")    cluster label of the extra data
    -m,--model   (default "./model.net")   the trained covnet
]]

print (opt)
local channel = 3
local height = 96
local width = 96


-- read in data and convert it to one tensor : data
print(c.blue '==>' ..' loading extra data')
local raw = torch.load('./stl-10/extra.t7b')
data = torch.FloatTensor(100000, channel, width, height)
local idx = 1
for i = 1, #raw.data[1] do
    data[idx]:copy(raw.data[1][i]:reshape(1, channel, width, height))
    idx = idx + 1
end


-- change color space and normalize the extra dataset
local normalization = nn.SpatialContrastiveNormalization(1,image.gaussian1D(1))
local mean_u = -3.1261832071966
local std_u = 12.419259853307
local mean_v = 1.4553072717114
local std_v = 15.0061438658
for i=1,data:size()[1] do
    xlua.progress(i, #raw.data[1])
    local rgb = data[i]
    local yuv = image.rgb2yuv(rgb)
    -- nomoralize y locally
    yuv[{1}] = normalization(yuv[{{1}}])
    data[i] = yuv
end
data:select(2,2):add(-mean_u)
data:select(2,2):div(std_u)
data:select(2,3):add(-mean_v)
data:select(2,3):add(std_v)


-- loading labels, put data into different groups
print(c.blue '==>' ..' loading labels')
local raw_l = torch.load(opt.labels)
print(raw_l:size())
clusters = {}
for i=1,10 do
    cluster[i] = {}
end
assert(#raw_l == data:size()[1], "label size not equal to sample size")
for i=1,raw_l:size()[1] do
    table.insert(cluster[i], data[i]:clone())
end

-- release memory
data = nil
collectgarbage()


-- loading model
print(c.blue '==>' ..' loading convnet Model')
local model = torch.load(opt.model)
model:evaluate()


-- make prediction on each cluser only pick up the class which has more than 
-- thresh number of samples.
result={
    data = {},
    labels = {}
}

for i=1,10 do
    local cluster = clusters[i]
    local thresh = math.ceil(0.6 * #cluster)
    local prediction = torch.FloatTensor(10, #cluster):zero()
    for j=1,#cluster do
        local output = model.forward(cluster[j])
        prediction[output][j] = 1
    end
    local pred_sum = torch.sum(prediction, 2)
    for j = 1,10 do
        if pred_sum >= thresh then
            for k = 1, #cluster do
                if prediction[j][k] == 1 then
                    table.insert(result.data, cluster[k])
                    table.insert(result.labels, j)
                end
            end
            break
        end
    end
end

print(#result.data)
torch.save("new_data.t7t", result)





