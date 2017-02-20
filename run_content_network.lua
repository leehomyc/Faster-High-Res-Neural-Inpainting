require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  gpu=1,
  overlapPred=4,
  model_file='models/imagenet_inpaintCenter.t7',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local function loadImage(path,size)
    local input = image.load(path, 3, 'float')
    input = image.scale(input,size,size)
    return input
end

modelG=util.load(opt.model_file,opt.gpu)

local real=torch.Tensor(7,3,512,512)
local real_ctx = torch.Tensor(7,3,128,128)
local fake2 = torch.Tensor(3,256,256)
local output=torch.Tensor(3,512,512)

for i=1,7 do
    real[i]=loadImage(string.format('examples/pink_%04d.jpg',i),512)
    real_ctx[i]:copy(image.scale(real[i],128,128))
end
fake = modelG:forward(real_ctx)
for i=1,7 do
    fake2:copy(image.scale(fake[i],256,256))
    output:copy(real[i])
    output[{{},{145,368},{145,368}}]:copy(fake2[{{},{17, 240},{17, 240}}])
    output[output:gt(1)]=1
    output[output:lt(-1)]=-1
    output[output:gt(1)]=1
    image.save(string.format('examples/fake_%04d.png',i),output)
end


