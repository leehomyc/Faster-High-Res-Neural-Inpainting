------------------------------------------------------------------------
-- mylbfgs
------------------------------------------------------------------------
function mylbfgs(opfunc, x, config, state, mask)
   -- get/update state
   local config = config or {}
   local state = state or config
   local maxIter = tonumber(config.maxIter) or 20
   local maxEval = tonumber(config.maxEval) or maxIter*1.25
   local tolFun = config.tolFun or 1e-5
   local tolX = config.tolX or 1e-9
   local nCorrection = config.nCorrection or 100
   local lineSearch = config.lineSearch
   local lineSearchOpts = config.lineSearchOptions
   local learningRate = config.learningRate or 1
   local isverbose = config.verbose or false
   

   state.funcEval = state.funcEval or 0
   state.nIter = state.nIter or 0
   -- verbose function
   local function verbose(...)
      if isverbose then print('<mylbfgs> ', ...) end
   end

   -- import some functions
   local zeros = torch.zeros
   local randn = torch.randn
   local append = table.insert
   local abs = math.abs
   local min = math.min

   -- evaluate initial f(x) and df/dx
   local f,g = opfunc(x)
   g:cmul(mask) -- add by chris 
   local f_hist = {f}
   local currentFuncEval = 1
   state.funcEval = state.funcEval + 1

   -- check optimality of initial point
   state.tmp1 = state.abs_g or zeros(g:size()); local tmp1 = state.tmp1
   tmp1:copy(g):abs()
   if tmp1:sum() <= tolFun then
      -- optimality condition below tolFun
      verbose('optimality condition below tolFun')
      return x,f_hist
   end

   -- variables cached in state (for tracing)
   local d = state.d
   local t = state.t
   local old_dirs = state.old_dirs
   local old_stps = state.old_stps
   local Hdiag = state.Hdiag
   local g_old = state.g_old
   local f_old = state.f_old

   -- optimize for a max of maxIter iterations
   local nIter = 0
   while nIter < maxIter do
      -- keep track of nb of iterations
      nIter = nIter + 1
      state.nIter = state.nIter + 1
      -- print(state.nIter)
      ------------------------------------------------------------
      -- compute gradient descent direction
      ------------------------------------------------------------
      if state.nIter == 1 then
        d = g:clone():mul(-1) -- -g
        old_dirs = {}
        old_stps = {}
        Hdiag = 1
      else
        -- do lbfgs update (update memory)
        local y = g:clone():add(-1, g_old)  -- g - g_old
        local s = d:clone():mul(t)  -- d*t
        local ys = y:dot(s)  -- y*s 
             
         if ys > 1e-10 then
            -- updating memory
            if #old_dirs == nCorrection then
               -- shift history by one (limited-memory)
               local prev_old_dirs = old_dirs
               local prev_old_stps = old_stps
               old_dirs = {}
               old_stps = {}
               for i = 2,#prev_old_dirs do
                  append(old_dirs, prev_old_dirs[i])
                  append(old_stps, prev_old_stps[i])
               end
            end

            -- store new direction/step
            append(old_dirs, s)
            append(old_stps, y)

            -- update scale of initial Hessian approximation
            Hdiag = ys / y:dot(y)  -- (y*y)

            -- cleanup
            collectgarbage()
         end

         -- compute the approximate (L-BFGS) inverse Hessian 
         -- multiplied by the gradient
         local p = g:size(1)
         local k = #old_dirs

         state.ro = state.ro or zeros(nCorrection); local ro = state.ro
         for i = 1,k do
            ro[i] = 1 / old_stps[i]:dot(old_dirs[i])
         end

         state.q = state.q or zeros(nCorrection+1,p):typeAs(g) 
         local q = state.q
         state.r = state.r or zeros(nCorrection+1,p):typeAs(g)
         local r = state.r
         state.al = state.al or zeros(nCorrection):typeAs(g)
         local al = state.al
         state.be = state.be or zeros(nCorrection):typeAs(g)
         local be = state.be

         q[k+1] = g:clone():mul(-1)  -- -g

         for i = k,1,-1 do
            al[i] = old_dirs[i]:dot(q[i+1]) * ro[i]
            q[i] = q[i+1]
            q[i]:add(-al[i], old_stps[i])
         end

         -- multiply by initial Hessian
         r[1] = q[1]:clone():mul(Hdiag)  -- q[1] * Hdiag

         for i = 1,k do
            be[i] = old_stps[i]:dot(r[i]) * ro[i]
            r[i+1] = r[i]
            r[i+1]:add((al[i] - be[i]), old_dirs[i])
         end

         -- final direction:
         d:copy(r[k+1])
      end -- end if state.nIter == 1 then

      g_old = g:clone()
      f_old = f

      ------------------------------------------------------------
      -- compute step length
      ------------------------------------------------------------
      -- directional derivative
      local gtd = g:dot(d)  -- g * d

      -- check that progress can be made along that direction
      if gtd > -tolX then
         break
      end

      -- reset initial guess for step size
      if state.nIter == 1 then
         tmp1:copy(g):abs()
         t = min(1,1/tmp1:sum()) * learningRate
      else
         t = learningRate
      end

      -- optional line search: user function
      local lsFuncEval = 0
      if lineSearch and type(lineSearch) == 'function' then
         -- perform line search, using user function
         f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
         append(f_hist, f)
      else
         -- no line search, simply move with fixed-step
         x:add(t,d)
         if nIter ~= maxIter then
            -- re-evaluate function only if not in last iteration
            -- the reason we do this: in a stochastic setting,
            -- no use to re-evaluate that function here
            f,g = opfunc(x)
            g:cmul(mask) -- add by chris
            lsFuncEval = 1
            append(f_hist, f)
         end
      end

      -- update func eval
      currentFuncEval = currentFuncEval + lsFuncEval
      state.funcEval = state.funcEval + lsFuncEval
      
      ------------------------------------------------------------
      -- check conditions
      ------------------------------------------------------------
      if nIter == maxIter then
         -- no use to run tests
         verbose('reached max number of iterations')
         break
      end

      if currentFuncEval >= maxEval then
         -- max nb of function evals
         verbose('max nb of function evals')
         break
      end

      tmp1:copy(g):abs()
      if tmp1:sum() <= tolFun then
         -- check optimality
         verbose('optimality condition below tolFun')
         break
      end

      tmp1:copy(d):mul(t):abs()
      if tmp1:sum() <= tolX then
         -- step size below tolX
         verbose('step size below tolX')
         break
      end

      if abs(f-f_old) < tolX then
         -- function value changing less than tolX
         verbose('function value changing less than tolX')
         break
      end
   end -- end while nIter < maxIter do

   -- save state
   state.old_dirs = old_dirs
   state.old_stps = old_stps
   state.Hdiag = Hdiag
   state.g_old = g_old
   state.f_old = f_old
   state.t = t
   state.d = d

   -- return optimal x, and history of f(x)
   return x,f_hist,currentFuncEval
end