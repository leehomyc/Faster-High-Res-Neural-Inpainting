local util = {}

-- Designed to handle TableContainers like ParallelTable() etc. as for
-- those m.output is a table; otherwise for Containers like Sequential()
-- etc. m.output is a tensor
-- NOTE: mAttribute may be call by reference or value depending on what it
-- is. So this function is written by assuming call by value setup.
local function recursiveTableClear(mAttribute, compulsoryAttr)
    compulsoryAttr = compulsoryAttr==nil and true or compulsoryAttr
    if type(mAttribute) == 'table' then
        for i=1,#mAttribute do
            mAttribute[i] = recursiveTableClear(mAttribute[i])
        end
        return mAttribute
    else
        if compulsoryAttr then
            return mAttribute.new()
        else
            return mAttribute and mAttribute.new() or nil
        end
    end
end
-- Designed to handle nested modules
-- NOTE: netsave will always be call by reference
local function recursiveSave(netsave)
    for k, l in ipairs(netsave.modules) do
        -- check if itself is a parent module
        if netsave.modules[k].modules~=nil then
            recursiveSave(netsave.modules[k])
        end

        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                          l.kW, l.kH, l.dW, l.dH, 
                          l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
                           l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = recursiveTableClear(m.output)
        m.gradInput = recursiveTableClear(m.gradInput)
        m.finput = recursiveTableClear(m.finput,false)
        m.fgradInput = recursiveTableClear(m.fgradInput,false)
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
        -- TODO: figure out why giant storage-offsets being created on typecast
        if m.weight then
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
end
function util.save(filename, net, gpu)
    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end
    recursiveSave(netsave)
    netsave.output = recursiveTableClear(netsave.output)
    netsave.gradInput = recursiveTableClear(netsave.gradInput)

    -- applies recursively to all containers as well
    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    -- for k, l in ipairs(netsave.modules) do
    --     print(k,l)
    -- end
    -- print(netsave.modules)
    -- print(netsave.output)
    -- print(netsave.gradInput)
    -- print(netsave)
    -- for k, l in pairs(netsave) do
    --     print(k,l)
    -- end

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
        m.gradWeight = m.weight:clone():zero();
        m.gradBias = m.bias:clone():zero(); end end)
   return net
end

local function recursiveCudnn(net)
    for k, l in ipairs(net.modules) do
        -- check if itself is a parent module
        if net.modules[k].modules~=nil then
            recursiveCudnn(net.modules[k])
        end

        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                         l.kW, l.kH, l.dW, l.dH,
                         l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
end
function util.cudnn(net)
    recursiveCudnn(net)
    return net
end

return util

