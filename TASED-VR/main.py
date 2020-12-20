from torchsummary import summary
from model import TASED_v2

import torch

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = TASED_v2()

    x = torch.rand((1, 3, 32, 224, 384))
    output = model(x)
    print(output.size())

    # summary(model, (1, 3, 32, 224, 384))