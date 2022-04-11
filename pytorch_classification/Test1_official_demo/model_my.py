import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        """
        Convolution Calculating: N = (W - F + 2P)/ S + 1
        N: input image size W*W
        F: Size of filter: F*F
        S: Stride
        P: Padding
        """
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120) # input of a nn.Linear FCN is a one-dimensional vector
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)           # output(16, 14, 14)
        x = F.relu(self.conv2(x))   # output(32, 10, 10)
        x = self.pool2(x)           # output(32, 5, 5)
        x = x.view(-1, 32*5*5)      # output(32*5*5)
        x = self.fc1(x)             # output(120)
        x = self.fc2(x)             # output(84)
        x = self.fc3(x)             # output(10)
        return x

'''
# debug model structure
# Run this code with:
python model_my.py
'''
if __name__ == '__main__':

    import torch
    input = torch.rand([32, 3, 32, 32])
    model = LeNet()
    print(model)
    output = model(input)
    print(output.shape)




