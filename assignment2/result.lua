-- result.lua take model.net, produce the prediction.csv

require 'torch'
require 'xlua'
require 'image'
require 'cutorch'
require 'cunn'
require 'nn'

opt = lapp[[
    -m,--model   (default "./model.net")   the model file used for evaluaiton
    -i,--input   (default "./stl-10/test.t7b")  the path to the test set
]]

------------------------------------------------------------
-- prepare the test data set
------------------------------------------------------------

print "load in the test data..."
local raw_test = torch.load(opt.input)
if not raw_test then
    print("input file not found")
    os.exit()
end

local testSize = 8000
local channel = 3
local width = 96
local height = 96

data = torch.FloatTensor(testSize, channel, width, height)
label = torch.FloatTensor(testSize)

local idx = 1
for i = 1,10 do
    for j = 1, #raw_test.data[i] do
        data[idx]:copy(raw_test.data[i][j])
        label[idx] = i
        idx = idx+1
    end
end

raw_test = nil
collectgarbage()

-- change color space and normalize it
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
local mean_u=-3.1318353592303
local mean_v=1.4737367561029
local std_v=15.019115155577
local std_u=12.420348900831
for i=1,data:size()[1] do
    xlua.progress(i, data:size()[1])
    local rgb = data[i]
    local yuv = image.rgb2yuv(rgb):double()
    -- normalize y locally
    yuv[{1}] = normalization(yuv[{{1}}])
    data[i] = yuv:float()
end
data:select(2,2):add(-mean_u)
data:select(2,2):div(std_u)
data:select(2,3):add(-mean_v)
data:select(2,3):div(std_v)
collectgarbage()



-----------------------------------------------------------
-- make prediction on test data
-----------------------------------------------------------
outf = io.open("prediction.csv", "w")
outf:write("Id,Prediction\n")
model = torch.load(opt.model)
model:evaluate()

local nCorrect = 0
local batchSize =25

for i=1, data:size()[1], batchSize do
    xlua.progress(i, data:size()[1])
    local lasti = math.min(data:size()[1], i+batchSize-1)
    local m = lasti - i + 1
    local t = torch.FloatTensor(m, 3, 96, 96):copy(data[{{i,lasti},{},{},{}}])
    local output = model:forward(t:cuda())
    local val, pred = torch.max(output, 2)
    for j=i,lasti do
        outf:write(j .. "," .. pred[j-i+1][1] .. "\n")
        if label[j] == pred[j-i+1][1] then
            nCorrect = nCorrect + 1
        end
    end
end
outf.close()

