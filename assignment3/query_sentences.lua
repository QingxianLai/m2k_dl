
stringx = require('pl.stringx')
require 'io'

function readline()
    local line = io.read("*line")
    if line = nil then error({code="EOF"}) end
    line = stringx.split(line)
    if tonumber(line[1]) == nil then error({code="EOF"}) end
    return line
end


while true do
    print("Query: len word1 word2 etc")
    local ok, line = pcall(readline)
    if not ok then
        print line
    else
        -- do something here
    end
end

