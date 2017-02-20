function computeMRF(input, size, stride, gpu, backend)
	local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride)
	local dim_1 = input:size()[1] * size * size
	local dim_2 = coord_y:nElement()
	local dim_3 = coord_x:nElement()
	local t_feature_mrf = torch.Tensor(dim_2 * dim_3, input:size()[1], size, size)

	if gpu >= 0 then
	  if backend == 'cudnn' then
	    t_feature_mrf = t_feature_mrf:cuda()
	  else
	    t_feature_mrf = t_feature_mrf:cl()
	  end
	end
	local count = 1
	for i_row = 1, dim_2 do
	  for i_col = 1, dim_3 do
	    t_feature_mrf[count] = input[{{1, input:size()[1]}, {coord_y[i_row], coord_y[i_row] + size - 1}, {coord_x[i_col], coord_x[i_col] + size - 1}}]
	    count = count + 1
	  end
	end
	local feature_mrf = t_feature_mrf:resize(dim_2 * dim_3, dim_1)

	return t_feature_mrf, feature_mrf, coord_x, coord_y
end


function computeMRFnoTensor(input, size, stride, gpu, backend)
	local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride)
	local dim_1 = input:size()[1] * size * size
	local dim_2 = coord_y:nElement()
	local dim_3 = coord_x:nElement()
	local t_feature_mrf = torch.Tensor(dim_2 * dim_3, input:size()[1], size, size)

	if gpu >= 0 then
	  if backend == 'cudnn' then
        t_feature_mrf = t_feature_mrf:cuda()
      else
        t_feature_mrf = t_feature_mrf:cl()
      end
	end
	local count = 1
	for i_row = 1, dim_2 do
	  for i_col = 1, dim_3 do
	    t_feature_mrf[count] = input[{{1, input:size()[1]}, {coord_y[i_row], coord_y[i_row] + size - 1}, {coord_x[i_col], coord_x[i_col] + size - 1}}]
	    count = count + 1
	  end
	end
	local feature_mrf = t_feature_mrf:resize(dim_2 * dim_3, dim_1)

    t_feature_mrf = nil
    collectgarbage()
	return feature_mrf, coord_x, coord_y
end


function drill_computeMRFfull(input, size, stride, gpu)
	local coord_x, coord_y = computegrid(input:size()[3], input:size()[2], size, stride, 1)
	local dim = torch.Tensor(2)
	return coord_x, coord_y
end


function sampleMRFAndTensorfromLocation2(coord_x, coord_y, input, size, gpu)
	local t_feature_mrf = torch.Tensor(coord_x:nElement(), input:size()[1], size, size)
	for i_patch = 1, coord_x:nElement() do
		t_feature_mrf[i_patch] = input[{{1, input:size()[1]}, {coord_y[i_patch], coord_y[i_patch] + size - 1}, {coord_x[i_patch], coord_x[i_patch] + size - 1}}]
	end
    local feature_mrf = t_feature_mrf:reshape(coord_x:nElement(), input:size()[1] * size * size)
	return t_feature_mrf, feature_mrf
end


function computeBB(width, height, alpha)
	local min_x, min_y, max_x, max_y
	local x1 = 1
	local y1 = 1
	local x2 = width
	local y2 = 1
	local x3 = width
	local y3 = height
	local x4 = 1
	local y4 = height
	local x0 = width / 2
	local y0 = height / 2

	local x1r = x0+(x1-x0)*math.cos(alpha)+(y1-y0)*math.sin(alpha)
	local y1r = y0-(x1-x0)*math.sin(alpha)+(y1-y0)*math.cos(alpha)

	local x2r = x0+(x2-x0)*math.cos(alpha)+(y2-y0)*math.sin(alpha)
	local y2r = y0-(x2-x0)*math.sin(alpha)+(y2-y0)*math.cos(alpha)

	local x3r = x0+(x3-x0)*math.cos(alpha)+(y3-y0)*math.sin(alpha)
	local y3r = y0-(x3-x0)*math.sin(alpha)+(y3-y0)*math.cos(alpha)

	local x4r = x0+(x4-x0)*math.cos(alpha)+(y4-y0)*math.sin(alpha)
	local y4r = y0-(x4-x0)*math.sin(alpha)+(y4-y0)*math.cos(alpha)

	-- print(x1r .. ' ' .. y1r .. ' ' .. x2r .. ' ' .. y2r .. ' ' .. x3r .. ' ' .. y3r .. ' ' .. x4r .. ' ' .. y4r)
	if alpha > 0 then
	  -- find intersection P of line [x1, y1]-[x4, y4] and [x1r, y1r]-[x2r, y2r]
	  local px1 = ((x1 * y4 - y1 * x4) * (x1r - x2r) - (x1 - x4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
	  local py1 = ((x1 * y4 - y1 * x4) * (y1r - y2r) - (y1 - y4) * (x1r * y2r - y1r * x2r)) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
	  local px2 = px1 + 1
	  local py2 = py1
	  -- print(px1 .. ' ' .. py1)
	  -- find the intersection Q of line [px1, py1]-[px2, py2] and [x2r, y2r]-[x3r][y3r]

	  local qx = ((px1 * py2 - py1 * px2) * (x2r - x3r) - (px1 - px2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))
	  local qy = ((px1 * py2 - py1 * px2) * (y2r - y3r) - (py1 - py2) * (x2r * y3r - y2r * x3r)) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))  
	  -- print(qx .. ' ' .. qy)

	  min_x = width - qx
	  min_y = qy
	  max_x = qx
	  max_y = height - qy
	else if alpha < 0 then
	  -- find intersection P of line [x2, y2]-[x3, y3] and [x1r, y1r]-[x2r, y2r]
	  local px1 = ((x2 * y3 - y2 * x3) * (x1r - x2r) - (x2 - x3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
	  local py1 = ((x2 * y3 - y1 * x3) * (y1r - y2r) - (y2 - y3) * (x1r * y2r - y1r * x2r)) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
	  local px2 = px1 - 1
	  local py2 = py1
	  -- find the intersection Q of line [px1, py1]-[px2, py2] and [x1r, y1r]-[x4r][y4r]
	  local qx = ((px1 * py2 - py1 * px2) * (x1r - x4r) - (px1 - px2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))
	  local qy = ((px1 * py2 - py1 * px2) * (y1r - y4r) - (py1 - py2) * (x1r * y4r - y1r * x4r)) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))  
	  min_x = qx
	  min_y = qy
	  max_x = width - min_x
	  max_y = height - min_y
	  else
	    min_x = x1
	    min_y = y1
	    max_x = x2
	    max_y = y3
	  end
	end

	return math.max(math.floor(min_x), 1), math.max(math.floor(min_y), 1), math.floor(max_x), math.floor(max_y)
end

function computegrid(width, height, block_size, block_stride, flag_all)
	local coord_block_y = torch.range(1, height - block_size + 1, block_stride)
	if flag_all == 1 then
		if coord_block_y[#coord_block_y] < height - block_size + 1 then
		  local tail = torch.Tensor(1)
		  tail[1] = height - block_size + 1
		  coord_block_y = torch.cat(coord_block_y, tail)
		end
	end
	local coord_block_x = torch.range(1, width - block_size + 1, block_stride)
	if flag_all == 1 then
		if coord_block_x[#coord_block_x] < width - block_size + 1 then
		  local tail = torch.Tensor(1)
		  tail[1] = width - block_size + 1
		  coord_block_x = torch.cat(coord_block_x, tail)
		end
	end
	return coord_block_x, coord_block_y
end

function preprocess(img)
	local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
	local perm = torch.LongTensor{3, 2, 1}
	img = img:index(1, perm):mul(256.0)
	mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
	img:add(-1, mean_pixel)
	return img
end

-- Undo the above preprocessing.
function deprocess(img)
	local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
	mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
	img = img + mean_pixel:float()
	local perm = torch.LongTensor{3, 2, 1}
	img = img:index(1, perm):div(256.0)
	return img
end

function run_tests(run_type, list_params)
	local wrapper = run_type
    for i_test = 1, #list_params do
        wrapper.run_test(table.unpack(list_params[i_test]))
    end
end