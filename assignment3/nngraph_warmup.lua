require 'torch'
require 'nn'
require 'nngraph'

x = nn.Identity()()
y = nn.Identity()()
l1_x = nn.Tanh()(nn.Linear(4,2)(x))
l1_y = nn.Sigmoid()(nn.Linear(5,2)(y))
z = nn.Identity()()
l2 = nn.CMulTable()({l1_x, l1_y})
a = nn.CAddTable()({l2, z})

g = nn.gModule({x,y,z},{a})


x_data = torch.rand(4)
y_data = torch.rand(5)
z_data = torch.rand(2)
gradOutput = torch.ones(2)

g:updateOutput({x_data, y_data, z_data})
g:updateGradInput({x_data, y_data, z_data}, gradOutput)

print(g:forward({x_data, y_data, z_data}))
