import torch
from PIL import Image
from model_my import LeNet
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


net = LeNet()
net.load_state_dict(torch.load('LeNet.pth'))

im = Image.open('download.jpeg')
print(im)
im = transform(im) # [C, H, W]
print(im.shape)
im = torch.unsqueeze(im, dim=0)
print(im.shape)

with torch.no_grad():
    outputs = net(im)
    print(outputs)
    a = torch.softmax(outputs, dim=1)
    print(a)
    b = torch.max(a, dim=1)[1]
    print('b', b)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horsd', 'ship', 'truch')
    predict = torch.max(outputs, dim=1)[1]
    print(predict)
    print(int(predict))


print(classes[int(predict)])

