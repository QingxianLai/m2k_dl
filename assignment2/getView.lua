
require 'torch'
require 'xlua'
require 'image'
require 'cutorch'
require 'cunn'
require 'nn'

opt = lapp[[
    -m,--model   (default "./model.net")    the model file
    -i,--input   (default "./stl-10/test.t7b")  input data
]]


print "load in the test data"
local raw_data = torch.load(opt.input)
if not raw_data then
    print("intput file not found")
    os.exit()
end

local channel = 3
local width = 96
local height = 96

if #raw_data.data == 10 then
    -- train, val, test 
    local size = #raw_data.data[1] * 10
    data = torch.FloatTensor(size, channel, width, height)
    label = torch.FloatTensor(size)
    
    local idx = 1
    for i = 1,10 do
        for j = 1,#raw_data.data[i] do
            data[idx]:copy(raw_data.data[i][j])
            label[idx] = i
            idx = idx + 1
        end
    end
    raw_data = nil

else
    -- extra
    local size = #raw_data.data[1]
    data = torch.FloatTensor(size, channel, width, height)
    label = nil
    for i=1,size do
        data[i]:copy(raw_data.data[1][i])
    end
    raw_data = nil
end

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


-----------------------------------------------------------
-- make prediction on test data
-----------------------------------------------------------
model = torch.load(opt.model)
model:evaluate()

local batchSize =25
local resultTensor = torch.FloatTensor(data:size()[1], 10)
local nCorrect = 25

for i=1, data:size()[1], batchSize do
    xlua.progress(i, data:size()[1])
    local lasti = math.min(data:size()[1], i+batchSize-1)
    local m = lasti - i + 1
    local t = torch.FloatTensor(m, 3, 96, 96):copy(data[{{i,lasti},{},{},{}}])
    local output = model:forward(t:cuda())
    local val,pred = torch.max(output,2)
    local lastView = model:get(54):get(6).output -- 10 
    --local lastView = model:get(53).output -- 4098
    resultTensor[{{i,lasti}, {}}]:copy(lastView)
    for j=i,lasti do
        if label[j] == pred[j-i+1][1] then
            nCorrect = nCorrect + 1
        end
    end

end

torch.save("last_layer_10.t7", resultTensor)
print(string.format("The test score is %s", (nCorrect/data:size()[1])))
