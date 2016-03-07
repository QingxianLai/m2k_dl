require 'xlua'
require 'nn'
require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
local c = require 'trepl.colorize'

opt = lapp[[
    -l,--labels (default "./labels.t7")    cluster label of the extra data
    -m,--model   (default "./model.net")   the trained covnet
]]

print (opt)
local channel = 3
local height = 96
local width = 96
local extraSize = 5000
-- load index
local index = torch.load('./index.t7')


-- read in data and convert it to one tensor : data
print(c.blue '==>' ..' loading extra data')
local raw = torch.load('./stl-10/extra.t7b')
data = torch.FloatTensor(extraSize, channel, width, height)
local idx = 1
for i = 1, extraSize do
    data[idx]:copy(raw.data[1][index[i]]:reshape(1, channel, width, height))
    idx = idx + 1
end

raw = nil
collectgarbage()


-- change color space and normalize the extra dataset
local normalization = nn.SpatialContrastiveNormalization(1,image.gaussian1D(1))
local mean_u = -3.1261832071966
local std_u = 12.419259853307
local mean_v = 1.4553072717114
local std_v = 15.0061438658
for i=1,data:size()[1] do
    xlua.progress(i, data:size()[1])
    local rgb = data[i]:double()
    local yuv = image.rgb2yuv(rgb)
    -- nomoralize y locally
    yuv[{1}] = normalization(yuv[{{1}}])
    data[i] = yuv:float()
end
data:select(2,2):add(-mean_u)
data:select(2,2):div(std_u)
data:select(2,3):add(-mean_v)
data:select(2,3):add(std_v)

collectgarbage()

-- loading labels, put data into different groups
print(c.blue '==>' ..' loading labels')
local raw_l = torch.load(opt.labels)
print(raw_l:size())
clusters = {}
for i=1,10 do
    clusters[i] = {}
end
assert((#raw_l)[1] == data:size()[1], string.format("label size %s not equal to sample size%s ", (#raw_l), data:size()[1]))
for i=1,raw_l:size()[1] do
    xlua.progress(i, raw_l:size()[1])
    table.insert(clusters[raw_l[i]], data[i]:clone())
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
    local thresh = math.ceil(0.3 * #cluster)
    local batchSize = 64
    local pred_all = torch.FloatTensor(#cluster)
    local pred_sum = torch.FloatTensor(10):zero()

    for j = 1,#cluster,batchSize do
        xlua.progress(j, #cluster)
        local lastj = math.min(#cluster, j+batchSize-1)
        local m = lastj - j + 1
        local t = torch.FloatTensor(m, 3, 96, 96)
        for k = 0,m-1 do
            t[k+1]:copy(cluster[j+k])
        end
        local output = model:forward(t:cuda())
        local val, pred = torch.max(output, 2)
        pred_all[{{j,lastj}}]:copy(pred)
    end

    for j = 1, #cluster do
        pred_sum[pred_all[j]] = pred_sum[pred_all[j]] + 1
    end
    print(pred_sum)
    
    for j = 1,10 do
        if pred_sum[j] >= thresh then
            for k = 1, #cluster do
                if pred_all[k] == j then
                    table.insert(result.data, cluster[k])
                    table.insert(result.labels, j)
                end
            end
            break
        end
    end
end
print(#result.data)
--torch.save("new_data.t7t", result)



