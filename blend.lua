require 'image'
local source=image.load('')
local tgt=image.load('')
local mask=image.load('')

local res=torch.cmul(src,mask)+torch.cmul(tgt,1-mask)
image.save('',res)