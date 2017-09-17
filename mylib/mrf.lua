local MRFMM, parent = torch.class('nn.MRFMM', 'nn.Module')

function MRFMM:__init()
   parent.__init(self)
   self.max_response = nil
end

function MRFMM:implement(mode, target_mrf, tensor_target_mrf, target_mrfnorm, source_x, source_y, input_size, response_size, nInputPlane, nOutputPlane, kW, kH, dW, dH, threshold_conf, strength, gpu_chunck_size_1, gpu_chunck_size_2, backend, gpu)
  self.target_mrf = target_mrf:clone()
  self.target_mrfnorm = target_mrfnorm:clone()
  self.source_x = source_x
  self.source_y = source_y
  self.input_size = input_size
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kW = kW
  self.kH = kH
  self.dW = dW
  self.dH = dH
  self.threshold_conf = threshold_conf
  self.strength = strength
  self.padW = padW or 0
  self.padH = padH or self.padW
  self.bias = torch.Tensor(nOutputPlane):fill(0)
  self.backend = backend
  self.gpu = gpu
  if self.gpu >= 0 then
    if self.backend == 'cudnn' then
      self.bias = self.bias:cuda()
    else
      self.bias = self.bias:cl()
    end
  end

  self.gradTO = torch.Tensor(input_size[1], input_size[2], input_size[3])
  self.gradTO_confident = torch.Tensor(input_size[2], input_size[3])
  self.response = torch.Tensor(response_size[1], response_size[2], response_size[3]) 
  self.mode = mode -- memory or speed
  self.gpu_chunck_size_1 = gpu_chunck_size_1
  self.gpu_chunck_size_2 = gpu_chunck_size_2
  self.tensor_target_mrfnorm = torch.repeatTensor(target_mrfnorm, 1, self.gpu_chunck_size_2, input_size[3] - (kW - 1))
  
  if self.mode == 'speed' then 
    if self.backend == 'cudnn' then
      self.target_mrf = self.target_mrf:cuda()
      self.target_mrfnorm = self.target_mrfnorm:cuda()
      self.tensor_target_mrfnorm = self.tensor_target_mrfnorm:cuda()
      self.gradTO = self.gradTO:cuda()
      self.gradTO_confident = self.gradTO_confident:cuda()
      self.response = self.response:cuda()
    else
      self.target_mrf = self.target_mrf:cl()
      self.target_mrfnorm = self.target_mrfnorm:cl()
      self.tensor_target_mrfnorm = self.tensor_target_mrfnorm:cl()
      self.gradTO = self.gradTO:cl()
      self.gradTO_confident = self.gradTO_confident:cl()
      self.response = self.response:cl()
    end
  end

end


local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
    print('not contiguous, make it so')
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
   self._gradOutput = self._gradOutput or gradOutput.new()
   self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
   gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function MRFMM:updateOutput(input)
    input = makeContiguous(self, input)
    self.output = input:clone()
    return self.output
end

function MRFMM:updateGradInput(input, gradOutput)

  input = makeContiguous(self, input)
  self.gradTO = self.gradTO:fill(0)
  self.gradTO_confident = self.gradTO_confident:fill(0) + 1e-10
  local source_mrf, x, y = computeMRFnoTensor(input:float(), self.kW, 1, self.mode == 'memory' and -1 or 1, self.backend)
  local source_mrfnorm = torch.Tensor(source_mrf:size()[1])

  if self.mode == 'speed' then
      if self.backend == 'cudnn' then
        source_mrfnorm = torch.sqrt(torch.sum(torch.cmul(source_mrf, source_mrf), 2)):resize(1, y:nElement(), x:nElement())
      else
        for i_source = 1, source_mrf:size()[1] do
          source_mrfnorm[i_source] = torch.sqrt(torch.sum(torch.cmul(source_mrf[i_source], source_mrf[i_source])))
        end
        source_mrfnorm = source_mrfnorm:resize(1, y:nElement(), x:nElement())
      end
  else
      source_mrfnorm = torch.sqrt(torch.sum(torch.cmul(source_mrf, source_mrf), 2)):resize(1, y:nElement(), x:nElement())
  end
  local tensor_source_mrfnorm = torch.repeatTensor(source_mrfnorm, self.gpu_chunck_size_1, 1, 1)
  if self.gpu >= 0 then
    if self.backend == 'cudnn' then
      tensor_source_mrfnorm = tensor_source_mrfnorm:cuda()
    else
      tensor_source_mrfnorm = tensor_source_mrfnorm:cl()
    end
  end
  local nOutputPlane_all = self.nOutputPlane -- hacked for memory safety
  local num_chunk = math.ceil(nOutputPlane_all / self.gpu_chunck_size_1) 
  
  for i_chunk = 1, num_chunk do
    local i_start = (i_chunk - 1) * self.gpu_chunck_size_1 + 1
    local i_end = math.min(i_start + self.gpu_chunck_size_1 - 1, nOutputPlane_all)

   
    self.weight = self.target_mrf[{{i_start, i_end}, {1, self.target_mrf:size()[2]}}]

    if self.mode == 'memory' then
      if self.gpu >= 0 then
        if self.backend == 'cudnn' then
          self.weight = self.weight:cuda()
        else
          self.weight = self.weight:cl()
        end
      end
    end
    self.nOutputPlane = i_end - i_start + 1

    local subBias = self.bias:sub(i_start, i_end)
    if self.gpu < 0 then
      self.finput = torch.Tensor()
      self.fgradInput = torch.Tensor()
    end

    input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      subBias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
    )
    local temp = self.output

    -- normalize w.r.t source_mrfnorm
    if i_chunk < num_chunk then
        temp = temp:cdiv(tensor_source_mrfnorm)
    else
        temp = temp:cdiv(tensor_source_mrfnorm[{{1, i_end - i_start + 1}, {1, temp:size()[2]}, {1, temp:size()[3]}}])
    end

    if self.mode == 'memory' then 
      temp = temp:float()
    end
    self.response[{{i_start, i_end}, {1, self.response:size()[2]}, {1, self.response:size()[3]}}] = temp
  end
 
  if self.max_response == nil or self.max_response:size()[2] ~= self.response:size()[2] then    
    local num_chunk_2 = math.ceil(self.response:size()[2] / self.gpu_chunck_size_2) 
    for i_chunk_2 = 1, num_chunk_2 do
      local i_start = (i_chunk_2 - 1) * self.gpu_chunck_size_2 + 1
      local i_end = math.min(i_start + self.gpu_chunck_size_2 - 1, self.response:size()[2])
      if i_chunk_2 < num_chunk_2 then
        self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}] = self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}]:cdiv(self.tensor_target_mrfnorm)
      else
        self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}] = self.response[{{1, self.response:size()[1]}, {i_start, i_end}, {1, self.response:size()[3]}}]:cdiv(self.tensor_target_mrfnorm[{{1, self.response:size()[1]}, {1, i_end - i_start + 1}, {1, self.response:size()[3]}}])
      end
    end
    
    local max_response, max_id = torch.max(self.response, 1)
    self.max_response = max_response
    self.max_id = max_id
  end

  source_mrf = source_mrf:resize(source_mrf:size()[1], self.nInputPlane, self.kW, self.kH)
  self.target_mrf = self.target_mrf:resize(self.target_mrf:size()[1], self.nInputPlane, self.kW, self.kH)
 
  for i_patch = 1, self.source_x:nElement() do
    local sel_response = self.max_response[1][self.source_y[i_patch]][self.source_x[i_patch]]
    if sel_response >= self.threshold_conf then
      local sel_idx = self.max_id[1][self.source_y[i_patch]][self.source_x[i_patch]]
      local source_idx = (self.source_y[i_patch] - 1) * x:nElement() + self.source_x[i_patch]        
      self.gradTO[{{1, self.nInputPlane}, {self.source_y[i_patch], self.source_y[i_patch] + self.kH - 1}, {self.source_x[i_patch], self.source_x[i_patch] + self.kW - 1}}]:add(self.target_mrf[sel_idx] - source_mrf[source_idx])
      self.gradTO_confident[{{self.source_y[i_patch], self.source_y[i_patch] + self.kH - 1}, {self.source_x[i_patch], self.source_x[i_patch] + self.kW - 1}}]:add(1)    
    end
  end
  
  self.gradTO:cdiv(torch.repeatTensor(self.gradTO_confident, self.nInputPlane, 1, 1))

  self.nOutputPlane = nOutputPlane_all
  self.target_mrf = self.target_mrf:resize(self.target_mrf:size()[1], self.nInputPlane * self.kW * self.kH)

  if gradOutput:size()[1] == input:size()[1] then
    if self.gpu >= 0 then
      if self.backend == 'cudnn' then
        self.gradInput = gradOutput:clone() + self.gradTO:cuda() * self.strength * (-1)
      else
        self.gradInput = gradOutput:clone() + self.gradTO:cl() * self.strength * (-1)
      end
    else
      self.gradInput = gradOutput:clone() + self.gradTO * self.strength * (-1)
    end
  else
    self.gradInput = self.gradTO * self.strength * (-1)
  end

  source_mrf = nil
  source_mrfnorm = nil
  tensor_source_mrfnorm = nil
  collectgarbage()
  return self.gradInput
end

function MRFMM:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type)
end
