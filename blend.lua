require 'image'
local source=image.load('')
local tgt=image.load('')
local mask=image.load('')

local res=source*mask/255+tgt*(255-mask)/255
image.save('',res)