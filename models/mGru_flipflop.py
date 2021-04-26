from taiyaki.activation import tanh
from taiyaki.layers import (
    Convolution_Quant, GruMod_Quant, Reverse, Serial, GlobalNormFlipFlop_Quant)


def network(insize=1, size=256, winlen=19, stride=2, alphabet_info=None):
    nbase = 4 if alphabet_info is None else alphabet_info.nbase

    return Serial([
        Convolution_Quant(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod_Quant(size, size)),
        GruMod_Quant(size, size),
        Reverse(GruMod_Quant(size, size)),
        GruMod_Quant(size, size),
        Reverse(GruMod_Quant(size, size)),
        GlobalNormFlipFlop_Quant(size, nbase),
    ])
'''model=network()
print(model)
#print(model.sublayers[0].conv.weight)
print(model.sublayers[0].conv.bias)
model.sublayers[0].conv.bias=quantize_tensor(model.sublayers[0].conv.bias)
print(model.sublayers[0].conv.bias)
print(model.sublayers[1].layer.cudnn_gru.weight_hh_l0)
print(model.sublayers[1].layer.cudnn_gru.weight_ih_l0)
print(model.sublayers[1].layer.cudnn_gru.bias_ih_l0)
print(model.sublayers[2].cudnn_gru.weight_hh_l0)
print(model.sublayers[2].cudnn_gru.weight_ih_l0)
print(model.sublayers[2].cudnn_gru.bias_ih_l0)
print(model.sublayers[3].layer.cudnn_gru.weight_hh_l0)
print(model.sublayers[3].layer.cudnn_gru.weight_ih_l0)
print(model.sublayers[3].layer.cudnn_gru.bias_ih_l0)
print(model.sublayers[4].cudnn_gru.weight_hh_l0)
print(model.sublayers[4].cudnn_gru.weight_ih_l0)
print(model.sublayers[4].cudnn_gru.bias_ih_l0)
print(model.sublayers[5].layer.cudnn_gru.weight_hh_l0)
print(model.sublayers[5].layer.cudnn_gru.weight_ih_l0)
print(model.sublayers[5].layer.cudnn_gru.bias_ih_l0)
print(model.sublayers[6].linear.weight)
print(model.sublayers[6].linear.bias)'''
