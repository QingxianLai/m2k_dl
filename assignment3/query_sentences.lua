stringx = require('pl.stringx')
require 'io'
require 'nngraph'
require 'base'


-- read in data and dictionaries
data = require('data')
_ = data.traindataset(20)
vocab_map = data.vocab_map
vocab_rmap = data.rmap


-- read std in
function readline()
    local line = io.read("*line")
    if line == nil then error({code="EOF"}) end
    line = stringx.split(line)
    if tonumber(line[1]) == nil then error({code="init"}) end
    num = tonumber(line[1])
    words = {}
    for i = 2,#line do
        words[i-1] = line[i]:lower()
    end
    return {num, words}
end


-- read in model file
model_file = "lstm_model.obj"
model = torch.load(model_file)



local function reset_state()
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * 2 do
            model.start_s[d]:zero()
        end
    end
end


batch_size = 20

local function encode(word)
    
    id = nil
    if vocab_map[word] ~= nil then
        id = vocab_map[word]
    else
        id = vocab_map["<unk>"]
    end
    local batch = torch.Tensor(batch_size):fill(id)
    return batch
end


local function multi(pred)
    return torch.multinomial(torch.exp(pred), 1, true)[1]
end


local function query(num, words)
    reset_state()
    g_disable_dropout(model.rnns)
    local result = {}
    
    g_replace_table(model.s[0], model.start_s)
    for i = 1, #words do
        local x = encode(words[i])
        local y = x
        _, model.s[1], pred = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
        result[i] = words[i]
        g_replace_table(model.s[0], model.s[1])
    end
    
    next_word = multi(pred[1])

    for i = 1,num do   
        x = torch.Tensor(batch_size):fill(next_word)
        local y = x
        _, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        next_word = multi(pred[1])
        result[i+#words] = vocab_rmap[next_word]
        g_replace_table(model.s[0], model.s[1])
    end
    g_enable_dropout(model.rnns)
    return result
end


-- std in out loop
while true do
    print("Query: len word1 word2 etc")
    local ok, line = pcall(readline)
    if not ok then
        if line.code == "EOF" then
            break
        elseif line.code == "init" then
            print("not a number")
        else
            print(line)
        end
    else
        print("sentence: ")
        num = line[1]
        words = line[2]
        extra = query(num, words)
        for i = 1, #extra do
            io.write(extra[i] .. " ")
        end
        io.write("\n\n")
    end
end
