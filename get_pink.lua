require 'image'

local function loadImage(path,size)
    local input = image.load(path, 3, 'float')
    
    input:mul(2):add(-1)
 --   print(input:size())
    return input
end

for i=1,32 do
    real=loadImage(string.format('examples/real_%04d.png',i),512)
    real[{{},{166, 345},{166,345}}] = 2*255/255.0 - 1.0
    test=(real+1)/2
    image.save(string.format('examples/input_%04d.png',i),test)
end