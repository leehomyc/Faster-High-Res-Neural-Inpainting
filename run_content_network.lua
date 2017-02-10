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
  batchSize=32,
  niter=40,
  fineSize=128,
  ntrain = math.huge, 
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=512,
  overlapPred=4,
  train_folder='/media/harryyang/New Volume/vision-harry/ILSVRC2015/Data/CLS-LOC/test_large_set',
  model_file='/media/harryyang/New Volume/models/inpaintCenter/imagenet_inpaintCenter.t7',
  result_path='/media/harryyang/New Volume/vision-harry/ILSVRC2015/Data/CLS-LOC/test_result_larger_new_model_1000'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads, 0, opt)

modelG=util.load(opt.model_file,opt.gpu)

local real_ctx = torch.Tensor(opt.batchSize,3,128,128)
local fake2 = torch.Tensor(opt.batchSize,3,opt.loadSize/2,opt.loadSize/2)
--local input_ctx = torch.Tensor(opt.batchSize,3,opt.loadSize/2,opt.loadSize/2)
local input = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize) --64 64
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

 --  for j=1,opt.batchSize do
 --     input_ctx[j]:copy(image.scale(real_ctx[j],128,128))
 --  end
  -- input_ctx=image.scale(real_ctx,128,128)
--   input_ctx:copy(real_ctx)
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
     -- if cnt>201 then
     --   break
     -- end
    end
end




