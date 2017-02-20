require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  dataset = 'folder', 
  batchSize=7,
  niter=40,
  fineSize=128,
  ntrain = math.huge, 
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=128,
  overlapPred=4,
  train_folder='',
  model_file='',
  result_path=''
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local function loadImage(path)
   local input = image.load(path, nc, 'float')
   input = image.scale(input, loadSize,loadSize)
   return input
end

modelG=util.load(opt.model_file,opt.gpu)

local real=torch.Tensor(opt.batchSize,3,128,128)
local real_ctx = torch.Tensor(opt.batchSize,3,128,128)

for i=1,7 dofile
  real[i]=loadImage(string.format('examples/pink_%04d.png',i)
  real_ctx[i]:copy(real[i])
  fake = modelG:forward(real_ctx)
   for j=1,opt.batchSize do
      fake2[j]:copy(image.scale(fake[j],256,256))
   end 
   input[{{},{},{145,368},{145,368}}]:copy(fake2[{{},{},{17, 240},{17, 240}}])
   test:copy(input[j])
   test[test:gt(1)]=1
          test[test:lt(-1)]=-1
          local fake_rgb=test
          fake_rgb=(fake_rgb+1)/2
      image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),fake_rgb)
end


local fake2 = torch.Tensor(opt.batchSize,3,opt.loadSize/2,opt.loadSize/2))
local input = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize) 
local real = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize)
local test = torch.Tensor(3, opt.loadSize, opt.loadSize)

cnt=1
for i = 1, opt.niter do
   real = data:getBatch()
   --real:copy(real_ctx)
   input:copy(real)
   for j=1,opt.batchSize do
      real_ctx[j]:copy(image.scale(input[j],128,128))
   end
   real_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*117.0/255.0 - 1.0
   real_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*104.0/255.0 - 1.0
   real_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*123.0/255.0 - 1.0

   fake = modelG:forward(real_ctx)
   for j=1,opt.batchSize do
      fake2[j]:copy(image.scale(fake[j],256,256))
   end 
   input[{{},{},{145,368},{145,368}}]:copy(fake2[{{},{},{17, 240},{17, 240}}])
   
    for j=1,opt.batchSize do
        test:copy(real[j])
          local real_rgb=test
          real_rgb=(real_rgb+1)/2
      image.save(string.format('%s/real_%04d.png',opt.result_path,cnt),real_rgb)
      test:copy(input[j])
          test[test:gt(1)]=1
          test[test:lt(-1)]=-1
          local fake_rgb=test
          fake_rgb=(fake_rgb+1)/2
      image.save(string.format('%s/fake_%04d.png',opt.result_path,cnt),fake_rgb)
      cnt=cnt+1
     
    end
end




