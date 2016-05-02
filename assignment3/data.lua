--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local trainfn = ptb_path .. "ptb.train.txt"
local testfn  = ptb_path .. "ptb.test.txt"
local validfn = ptb_path .. "ptb.valid.txt"

local vocab_idx = 0
local vocab_map = {}

-- it does:
-- read in the file, raplace \n with '<eos>'
-- build a dictionary that maps each word to an index
-- then return the documents in the form of indexes(a one dimensional array
-- of indexes)
local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
        end
        x[i] = vocab_map[data[i]]
    end
    return x
end


-- reshape the 1-D input into round(x_inp:size(1)/batch_size) by batch_size
-- matrix, sentences goes in columns. 
--
-- x:sub(a1,a2,b1,b2) means select a view of indexes [a1:a2, b1:b2] on x
local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1
        local finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

-- load train data, reshape into batches
local function traindataset(batch_size)
   local x = load_data(trainfn)
   x = replicate(x, batch_size)
   return x
end
-- print(traindataset(20)[{{1,50},3}])

-- same with training data, loading and reshaping
local function validdataset(batch_size)
    local x = load_data(validfn)
    x = replicate(x, batch_size)
    return x
end


-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
-- data from n x 1 to n x batch_size
-- expand will copy the value
local function testdataset(batch_size)
    if testfn then
        local x = load_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
        return x
    end
end


return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        vocab_map=vocab_map}
