-- this script assume the data.lua in the same directory.
-- print the perplexity of test data 

require 'nngraph'
require 'base'
require 'xlua'


opt = lapp[[
    -m,--model   (default "./lstm_model.obj")   the model file used for evaluaiton
]]


ptb = require 'data'
_ = ptb.traindataset(20)

batch_size = 20
state_test = {data=ptb.testdataset(batch_size)}


-- load model
model_file = opt.model
model = torch.load(model_file)


function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 4 do
            model.start_s[d]:zero()
        end
    end
end


-- test model
reset_state(state_test)
g_disable_dropout(model.rnns)
local perp = 0
local len = state_test.data:size(1)

-- no batching here
g_replace_table(model.s[0], model.start_s)
for i = 1, (len - 1) do
    xlua.progress(i, len)
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
end
print(" ")
print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
g_enable_dropout(model.rnns)
