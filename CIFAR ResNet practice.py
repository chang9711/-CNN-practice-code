import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

batch_size = 4
learning_rate = 0.001
epoch_num = 100
layer_num = 2


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = dset.CIFAR10(root='./Desktop', train=True,download=False,transform=transform)
test_set = dset.CIFAR10(root='./Desktop', train=False,download=False,transform=transform)

train_set_load = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_set_load = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class BasicBlock(nn.Module):
    exp = 1

    def __init__ (self,in_plane,plane,stride=1):
        super(BasicBlock,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_plane,plane,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(plane, plane,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_plane != self.exp*plane:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_plane,plane,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(plane),
                nn.ReLU()
            )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += self.shortcut(x)
        return out

class Bottleneck(nn.Module):
    exp = 4

    def __init__ (self,in_plane,plane,stride=1):
        super(Bottleneck,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_plane,plane,kernel_size=1,bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(plane,plane,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(plane,self.exp*plane,kernel_size=1,bias=False),
            nn.BatchNorm2d(self.exp*plane),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_plane != self.exp * plane:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_plane, self.exp * plane, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.exp * plane),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += self.shortcut(x)
        return out

class ResNet(nn.Module):

    def __init__(self,block,num_blocks,num_class=10):
        self.in_plane=16
        super(ResNet,self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block, 16,num_blocks[0],stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 128, num_blocks[3], stride=2)
        self.fc = nn.Linear(128*block.exp,num_class)

    def make_layer(self,block,plane,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_plane,plane,stride=stride))
            self.in_plane = block.exp*plane
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2])

'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
loss_temp = 3
for epoch in range(epoch_num):
    running_loss = 0.0
    for i ,(images,labels) in enumerate(train_set_load):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

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
'''
cnn=ResNet18()
cnn.load_state_dict(torch.load('cifar/cifar10.pth'))
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
