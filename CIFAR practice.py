import torch
import torch.nn as nn
import torchvision.utils as util
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


batch_size = 4
learning_rate = 0.001
epoch_num = 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = dset.CIFAR10(root='./Desktop', train=True,download=False,transform=transform)
test_set = dset.CIFAR10(root='./Desktop', train=False,download=False,transform=transform)

train_set_load = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_set_load = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
def imshow(img):
    img = img / 2 + 0.5    # [-1,1] -> [0,1]
    npimg = img.numpy() # numpy 없이도 가능
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_set_load)
images, labels = dataiter.next()
imshow(util.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
'''

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
loss_temp = 3

for epoch in range(epoch_num):
    running_loss = 0.0
    for i ,(images,labels) in enumerate(train_set_load):
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = cnn(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()



        if (i+1)%2000 == 0:
            print('[%d/%d] loss: %.3f'%(epoch+1, i+1, running_loss/2000))
            if loss_temp >= running_loss/2000:
                loss_temp = running_loss/2000
                path = 'cifar/cifar10.pth'
                torch.save(cnn.state_dict(), path)
            print(loss_temp)
            running_loss = 0.0

print('finish!')



cnn.eval()

correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_set_load:
        images = Variable(images)
        outputs = cnn(images)
        _,predict = torch.max(outputs.data,1)
        correct += (predict==labels).sum()
        total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%'%(100*correct/total))

class_correct = list(0 for i in range(10))
class_total =list(0 for i in range(10))

with torch.no_grad():
    for images, labels in test_set_load:
        outputs = cnn(images)
        _, predict = torch.max(outputs,1)
        c = (predict == labels)
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%'%(classes[i],100*class_correct[i]/class_total[i]))











