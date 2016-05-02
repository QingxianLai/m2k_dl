-- to understand:
-- pcall means protected call, which will call a funciton and manage the error
-- it first return whether there is an error raised while calling, then it will 
-- return the function's return value
--
-- io.read("*line") means read teh next line
--
-- error(message) is raising the error with the message. 
--
-- pl.stringx is python style string operator

stringx = require('pl.stringx')
require 'io'

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if line[i] ~= 'foo' then error({code="vocab", word = line[i]}) end
  end
  return line
end

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    print("Thanks, I will print foo " .. line[1] .. " more times")
    for i = 1, line[1] do io.write('foo ') end
    io.write('\n')
  end
end
