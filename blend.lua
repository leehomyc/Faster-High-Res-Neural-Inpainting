require 'image'

for i=1,32 do
    src=image.load(string.format('examples/input_%04d.png',i))
    tgt=image.load(string.format('examples/demo_input_%04d.pngTOfake_%04d.png.jpg',i,i))
    mask=image.load('mask.png')
    mask=mask[{{1,3},{},{}}]
    res=torch.cmul(src,mask)+torch.cmul(tgt,1-mask)
    image.save(string.format('examples/result_%04d.png',i),res)
end
