------------------------------------------------------------------------
-- ContentLoss
------------------------------------------------------------------------
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')
function ContentLoss:__init(strength, target, normalize)
   parent.__init(self)
   self.strength = strength
   self.target = target
   self.normalize = normalize or false
   self.loss = 0
   self.crit = nn.MSECriterion()
end
function ContentLoss:updateOutput(input)
   if input:nElement() == self.target:nElement() then
      self.loss = self.crit:forward(input, self.target) * self.strength
   else
      -- print(input:size())
      -- print(self.target:size())    
      -- print('WARNING: Skipping content loss')
   end
   self.output = input
   return self.output
end
function ContentLoss:updateGradInput(input, gradOutput)
   if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
   end
   if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
   end
   self.gradInput:mul(self.strength)
   self.gradInput:add(gradOutput) 
   return self.gradInput
end

function ContentLoss:update(other)
   self.strength = other.strength
   self.target = other.target
   self.normalize = other.normalize
   self.loss = other.loss
   self.crit = other.crit
end