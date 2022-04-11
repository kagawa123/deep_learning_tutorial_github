import torch
import torchvision
import torch.nn as nn
from model_my import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=4)

print(type(testloader))
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()


"""
Visulization
"""
# classes = ('plane', 'car', 'bird', 'cat',
#          'deer', 'dog', 'frog', 'horsd', 'ship', 'truch')
#
# def imshow(img):
#     img = img /2 +0.5 # unnormalize
#     npimg = img.numpy() # convert tensor (C H W) to numpy (H W C)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print('        '.join('%s' % classes[test_label[k]] for k in range(5)))
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),
                       lr=0.001)

for epoch in range(5): # loop over the dataset multiple times

    running_loss = 0.0
    for step, data in enumerate(trainloader, 0): # loop over all the data in a dataset
        # get the inputs; data is a list of [inputs, lalbels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print
        running_loss += loss.item()
        if step % 500 == 499: # print every 500 mini-batches(32)
            with torch.no_grad():
                outputs = net(test_image) # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item()/test_label.size(0)

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss/500, accuracy))
                running_loss = 0.0
    print('next epoch')
print('Finishing Training')
#
# save_path = './LeNet.pth'
# torch.save(net.state_dict(), save_path)
#






