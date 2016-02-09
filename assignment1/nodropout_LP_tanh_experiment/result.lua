-- result.lua take model.net, produce the prediction.csv 


print '==> downloading dataset'

-- Here we download dataset files. 

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = 'mnist.t7'
--train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(test_file)  then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

----------------------------------------------------------------------
-- test size
tesize = 10000

----------------------------------------------------------------------
print '==> loading dataset'

--loaded = torch.load(train_file, 'ascii')

---- split the training set to training and validation
--train_size = trsize * 0.8
--valid_size = trsize - train_size

--trainData = {
   --data = loaded.data[{{1,train_size},{},{},{}}],
   --labels = loaded.labels[{{1,train_size},{},{},{}}],
   --size = function() return train_size end   -- TODO why use function
--}

loaded = torch.load(test_file, 'ascii')
testSet = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

testSet.data = testSet.data:float()

print '==> read in the mean and std'
ff = io.open("mean_n_std", "r")
mean = ff:read()
std = ff:read()
ff:close()
--mean = trainData.data[{ {},1,{},{} }]:mean()
--std = trainData.data[{ {},1,{},{} }]:std()
--trainData.data[{ {},1,{},{} }]:add(-mean)
--trainData.data[{ {},1,{},{} }]:div(std)

-- Normalize test data, using the training means/stds
testSet.data[{ {},1,{},{} }]:add(-mean)
testSet.data[{ {},1,{},{} }]:div(std)

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

--trainMean = trainData.data[{ {},1 }]:mean()
--trainStd = trainData.data[{ {},1 }]:std()

testMean = testSet.data[{ {},1 }]:mean()
testStd = testSet.data[{ {},1 }]:std()

--print('training data mean: ' .. trainMean)
--print('training data standard deviation: ' .. trainStd)

print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)

---------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   --if average then
      --cachedparams = parameters:clone()
      --parameters:copy(average)
   --end

   model = torch.load("model.net")
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
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
      --if opt.type == 'double' then input = input:double()
      --elseif opt.type == 'cuda' then input = input:cuda() end
      --local target = testSet.labels[t]

      -- test sample
      local pred = model:forward(input)
      --print(pred:size()[1])
      --print(type(pred))
      --print(testSet.labels[t])
        
      local max_pred = -10000
      local max_indx = 1
      for i = 1,pred:size()[1] do
          if pred[i] > max_pred then
              max_pred = pred[i]
              max_indx = i
          end
      end

      --if max_indx == 10 then max_indx = 0 end

      outf:write(t .. "," .. max_indx .. "\n")
      --confusion:add(pred, target) -- TODO  where does this confusion come from?
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
