require 'nn'
require 'nngraph'

local x = nn.Identity()()
local y = nn.Identity()()
local z = nn.Identity()()

local l1_x = nn.Tanh()(nn.Linear(4,2)(x))
local l1_y = nn.Sigmoid()(nn.Linear(5,2)(y))

local sl1_x = nn.Square()({l1_x})
local sl1_y = nn.Square()({l1_y})

local l2 = nn.CMulTable()({sl1_x,sl1_y})

local a = nn.CAddTable()({l2, z})

local g = nn.gModule({x,y,z},{a})


local x_data = torch.rand(4)
local y_data = torch.rand(5)
local z_data = torch.rand(2)
local gradOutput = torch.ones(2)

--g:updateOutput({x_data, y_data, z_data})
--g:updateGradInput({x_data, y_data, z_data}, gradOutput)

-- demo
print("x=")
print(x_data)
print("y=")
print(y_data)
print("z=")
print(z_data)

print("Forward Propagating:")
print(g:forward({x_data, y_data, z_data}))

print("GradOutput:")
print(gradOutput)

print("back propagate w.r.t. x")
print(g:backward({x_data, y_data, z_data}, gradOutput)[1])
print("back propagate w.r.t. y")
print(g:backward({x_data, y_data, z_data}, gradOutput)[2])
print("back propagate w.r.t. z")
print(g:backward({x_data, y_data, z_data}, gradOutput)[3])
