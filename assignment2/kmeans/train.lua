require 'xlua'
require 'optim'
require 'unsup'
local c = require 'trepl.colorize'

opt = lapp[[
   -d,--data                   (default "train")      input dataset
   -s,--save                  (default "logs")        subdirectory to save logs
   -b,--batchSize             (default 64)            batch size
   -k,--kclass                (default 10)            number of clusters
   -m,--model                    (default kmeans)        model name
   -i,--max_iter                 (default 100)           maximum number of iterations
]]

print(opt)

local channel = 3
local height = 96
local width = 96

print(c.blue '==>' ..' loading data')
if opt.data == 'train' then
    local raw = torch.load("./stl-10/train.t7b")
    data = torch.FloatTensor(4000, channel * width * height)
    local idx = 1
    for i = 1,#raw.data do
        for j = 1,#raw.data[i] do 
            data[idx]:copy(raw.data[i][j]:reshape(1, channel * width * height))
            idx = idx + 1
        end
    end
elseif opt.data == 'extra' then
    local raw = torch.load('./stl-10/extra.t7b')
    data = torch.FloatTensor(100000, channel * width * height)
    local idx = 1
    for i = 1, #raw.data[1] do
        data[idx]:copy(raw.data[1][i]:reshape(1, channel * width * height))
        idx = idx + 1
    end
end

data = data:float()

local k = opt.kclass

print(data:size())
centroids,labels,totalcounts = unsup.kmeans(data, k, opt.max_iter, opt.batchSize)

print(totalcounts)
torch.save("labels.t7", labels)
