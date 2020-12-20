# -*- encoding: utf-8 -*-
# @Time: 2020/09/11 16:14
# @Author: librah
# @Description: convert the pytorch model to torch script
# @File: convert_model.py
# @Version: 1.0

import torch
from model import TASED_v2

MODEL_WEIGHTS = './TASED_updated.pt'
CONVERT_WEIGTHS = './convert_weights_gpu.pt'

def convert_model():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TASED_v2().to(device)
    weight_dict = torch.load(MODEL_WEIGHTS,map_location=device)
    model_dict = model.state_dict()

    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print (' size? ' + name, param.size(), model_dict[name].size())
        else:
            print (' name? ' + name)
    
    model.eval()

    inputs = torch.rand(1, 3, 32, 224, 384).to(device)
    script_model = torch.jit.trace(model, inputs)

    output = model(torch.ones(1, 3, 32, 224, 384).to(device))
    print(output.size())

    script_model.save(CONVERT_WEIGTHS)

if __name__ == '__main__':
    convert_model()
    

