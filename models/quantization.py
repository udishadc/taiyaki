import torch
from torch import nn
import torch.quantization
import copy
import os
import time

torch.manual_seed(29592)

model_lstm = torch.load('mLstm_flipflop_model_r941_DNA.checkpoint')
qmodel_lstm = torch.quantization.quantize_dynamic(model_lstm, {nn.LSTM}, dtype=torch.qint8)


print('Here is the floating point version of this module:')
print(model_lstm)
print('')
print('and now the quantized version:')
print(qmodel_lstm)


def size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' ','Size :', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=size(model_lstm,"Original model")
q=size(qmodel_lstm,"Quantized model")
print("{0:.2f} times smaller".format(f/q))


