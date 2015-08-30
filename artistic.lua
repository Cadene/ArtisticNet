require 'torch'
require 'image'
require 'nn'
posix = require 'posix' -- if not getpid doesn't work...
require 'src/ArtistContentCriterion'
require 'src/SpatialArtisticConvolution'

------------------------------------------------------------------------
-- Command line arguments

local cmd = torch.CmdLine()
cmd:text()
cmd:text('ArtisticNet')
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1337, 'seed')
cmd:option('-threads', 4, 'threads')
cmd:option('-layer', 6, '[1,6]')

cmd:text()
opt = cmd:parse(arg or {})

------------------------------------------------------------------------
-- Global Effects

local pid = posix.getpid("pid")
print("# ... lunching using pid = "..pid)

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)

------------------------------------------------------------------------
-- Model

model = nn.Sequential()

-- SpatialArtisticConvolution is currently a simple ConvMM

-- 18,916,480
model:add(nn.SpatialArtisticConvolution(3, 96, 7, 7, 2, 2))
model:add(nn.ReLU(true))
model:add(nn.SpatialAveragePooling(3, 3, 3, 3))
model:add(nn.SpatialArtisticConvolution(96, 256, 7, 7, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialAveragePooling(2, 2, 2, 2))
model:add(nn.SpatialArtisticConvolution(256, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialArtisticConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialArtisticConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialArtisticConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- classifier
-- model:add(nn.SpatialAveragePooling(3, 3, 3, 3))
-- model:add(lf.SpatialConvolution(opt, 1024, 4096, 5, 5, 1, 1))
-- model:add(nn.ReLU(true))
-- model:add(nn.Dropout(opt.dropout))
-- model:add(lf.SpatialConvolution(opt, 4096, 4096, 1, 1, 1, 1))
-- model:add(nn.ReLU(true))
-- model:add(nn.Dropout(opt.dropout))
-- model:add(lf.SpatialConvolution(opt, 4096, nb_class, 1, 1, 1, 1))
-- model:add(nn.View(nb_class))
-- model:add(nn.LogSoftMax())

local m = model.modules
local ParamBank = require 'ParamBank'
local label = require 'overfeat_label'
local offset = 0
ParamBank:init("net_weight_1")
ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
ParamBank:read(    14112, {96},            m[offset+1].bias)
ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
ParamBank:read(  1218432, {256},           m[offset+4].bias)
ParamBank:read(  1218688, {512,256,3,3},   m[offset+7].weight)
ParamBank:read(  2398336, {512},           m[offset+7].bias)
ParamBank:read(  2398848, {512,512,3,3},   m[offset+9].weight)
ParamBank:read(  4758144, {512},           m[offset+9].bias)
ParamBank:read(  4758656, {1024,512,3,3},  m[offset+11].weight)
ParamBank:read(  9477248, {1024},          m[offset+11].bias)
ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+13].weight)
ParamBank:read( 18915456, {1024},          m[offset+13].bias)
-- classifier
-- ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+16].weight)
-- ParamBank:read(123774080, {4096},          m[offset+16].bias)
-- ParamBank:read(123778176, {4096,4096,1,1}, m[offset+18].weight)
-- ParamBank:read(140555392, {4096},          m[offset+18].bias)
-- ParamBank:read(140559488, {1000,4096,1,1}, m[offset+20].weight)
-- ParamBank:read(144655488, {1000},          m[offset+20].bias)

------------------------------------------------------------------------
-- Prepare image -> 3*221*221

prepare = function (path2img, dim_in, dim_out)
    local dim     = dim_in or 221
    local dim_out = dim_out or 221
    local img_dim
    local img_raw = image.load(path2img) -- [0,1] -> [0,255]img
    local rh = img_raw:size(2)
    local rw = img_raw:size(3)

    -- rescale to 3 * 256 * 256
    if rh < rw then
       rw = math.floor(rw / rh * dim)
       rh = dim
    else
       rh = math.floor(rh / rw * dim)
       rw = dim
    end
    local img_scale = image.scale(img_raw, rw, rh)

    local offsetx = 1
    local offsety = 1
    if rh < rw then
        offsetx = offsetx + math.floor((rw-dim)/2)
    else
        offsety = offsety + math.floor((rh-dim)/2)
    end
    img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]

    return img
end

------------------------------------------------------------------------
-- Main

output_layer = {2, 5, 8, 10, 12, 14}
l = output_layer[opt.layer]

function remove(self, index)
   index = index or #self.modules
   if index > #self.modules or index < 1 then
      error"index out of range"
   end
   table.remove(self.modules, index)
   if #self.modules > 0 then
       self.output = self.modules[#self.modules].output
       self.gradInput = self.modules[1].gradInput
   else
       self.output = torch.Tensor()
       self.gradInput = torch.Tensor()
   end
end

local size = #model.modules
print(l, size)
for i = l, size-1 do
    remove(model)
end

print(model)

criterion = nn.ArtistContentCriterion()

input_origin = prepare('data/fraise.jpg')
-- input_gener  = prepare('data/fraise.jpg')
input_gener  = torch.rand(3,221,221)

activ_origin = model:forward(input_origin):clone()

local alpha = -0.02
for i = 1, 50 do
    activ_gener  = model:forward(input_gener):clone()
    print('activ_gener', activ_gener:size())
    loss = criterion:forward(activ_origin, activ_gener)
    print('loss', loss)
    dloss_dag = criterion:backward(activ_origin, activ_gener)
    print('dloss_dag', dloss_dag:size())
    g = model:updateGradInput(input_gener, dloss_dag)
    print('g', g:size())
    input_gener:add(alpha, g)--:mul(0.5/torch.abs(g):mean()))
    image.save('rslt/fraise3_layer'..l..'_final.jpg', input_gener)
end







