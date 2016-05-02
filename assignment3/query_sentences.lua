stringx = require('pl.stringx')
require 'io'

-- read in data and dictionaries
data = require('data')
_ = data.traindataset(20)


-- read std in
function readline()
    local line = io.read("*line")
    if line == nil then error({code="EOF"}) end
    line = stringx.split(line)
    if tonumber(line[1]) == nil then error({code="EOF"}) end
    return line
end


-- read in model file
model = torch.load("model.obj")


-- std in out loop
while true do
    print("Query: len word1 word2 etc")
    local ok, line = pcall(readline)
    if not ok then
        print(line)
    else
        -- do something here
    end
end


function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end
