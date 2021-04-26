import torch
from torch import nn
import torch.nn.utils.prune as prune

model = torch.load('mGru_flipflop_remapping_model_r9_DNA.checkpoint')
#print(list(model.named_parameters()))
for name, param in model.named_parameters():
    if name in ['sublayers.5.layer.cudnn_gru.weight_hh_l0']:
        #print (param)
        #w = param
        #w =  w * (w > 0.1) 
        #param.data = w.data
        rows = param.shape[0]
        cols = param.shape[1]
        zeroes = 0
        for x in range(0, rows):
            for y in range(0, cols):
              #print(param.data[x,y].numpy()[0])
              if(param.data[x,y] < 0.18 and param.data[x,y] > -0.14):
                  param.data[x,y] = 0
                  zeroes = zeroes+1
              #if(param.data[x,y] > -0.3 and param.data[x,y] < 0):
              #    param.data[x,y] = 0
              #    zeroes = zeroes+1
              #if(param.data[x,y] < 0):
              #    param.data[x,y] = 0
              #    zeroes = zeroes+1
print((zeroes*100)/(rows*cols))

torch.save(model, './mGru_flipflop_remapping_model_r9_DNA_pruned.checkpoint')
model = torch.load('mGru_flipflop_remapping_model_r9_DNA_pruned.checkpoint')
#print(list(model.named_parameters()))
for name, param in model.named_parameters():
    if name in ['sublayers.5.layer.cudnn_gru.weight_hh_l0']:
        #print (param)
        weight = param.tolist()
        #for i in range(len(weight)):
           #print(*weight[i],sep="\n") 

