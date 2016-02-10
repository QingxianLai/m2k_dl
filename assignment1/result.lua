-- result.lua take model.net, produce the prediction.csv 
-- team: m2k_dl


print '==> downloading dataset'
-- Here we download dataset files. 
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(test_file)  then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end


print '==> loading dataset'
tesize = 10000 -- test size
loaded = torch.load(test_file, 'ascii')
testSet = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}


print '==> preprocessing data'
testSet.data = testSet.data:float()

-- read in the mean std value
ff = io.open("mean_n_std", "r")
mean = ff:read()
std = ff:read()
ff:close()

-- normalize the test data
testSet.data[{ {},1,{},{} }]:add(-mean)
testSet.data[{ {},1,{},{} }]:div(std)


print '==> verify statistics'

testMean = testSet.data[{ {},1 }]:mean()
testStd = testSet.data[{ {},1 }]:std()

print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)


---------------------------------------------------------------------
-- make prediction on test dataset
---------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
-- require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   model = torch.load("model.net")
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   outf = io.open("prediction.csv", "w")
   outf:write("Id,Prediction\n")
   for t = 1,testSet:size() do
      -- disp progress
      xlua.progress(t, testSet:size())

      -- get new sample
      local input = testSet.data[t]
      input = input:double()

      -- test sample
      local pred = model:forward(input)
        
      local max_pred = -10000
      local max_indx = 1
      for i = 1,pred:size()[1] do
          if pred[i] > max_pred then
              max_pred = pred[i]
              max_indx = i
          end
      end

      outf:write(t .. "," .. max_indx .. "\n")
   end
   outf:close()

   -- timing
   time = sys.clock() - time
   time = time / testSet:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix

   -- update log/plot
   --testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   --if opt.plot then
      --testLogger:style{['% mean class accuracy (test set)'] = '-'}
      --testLogger:plot()
   --end

   ---- averaged param use?
   --if average then
      ---- restore parameters
      --parameters:copy(cachedparams)
   --end
   
   ---- next iteration:
   --confusion:zero()
end
test()
