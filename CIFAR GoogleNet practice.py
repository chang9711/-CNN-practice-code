import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

batch_size = 4
learning_rate = 0.001
epoch_num = 20
layer_num = 2


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = dset.CIFAR10(root='./Desktop', train=True,download=False,transform=transform)
test_set = dset.CIFAR10(root='./Desktop', train=False,download=False,transform=transform)

train_set_load = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_set_load = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Inception(nn.Module):
    def __init__(self,in_plane,con1x1,red3x3,con3x3,red5x5,con5x5,pool_plane):
        super(Inception,self).__init__()

        self.incep1 = nn.Sequential(
            nn.Conv2d(in_plane,con1x1,kernel_size=1),
            nn.BatchNorm2d(con1x1),
            nn.ReLU()
        )
        self.incep2 = nn.Sequential(
            nn.Conv2d(in_plane, red3x3, kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(),
            nn.Conv2d(red3x3,con3x3,kernel_size=3,padding=1),
            nn.BatchNorm2d(con3x3),
            nn.ReLU()
        )
        self.incep3 = nn.Sequential(
            nn.Conv2d(in_plane, red5x5, kernel_size=1),
            nn.BatchNorm2d(red5x5),
            nn.ReLU(),
            nn.Conv2d(red5x5,con5x5,kernel_size=3,padding=1),
            nn.BatchNorm2d(con5x5),
            nn.ReLU(),
            nn.Conv2d(con5x5, con5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(con5x5),
            nn.ReLU()
        )
        self.incep4 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_plane, pool_plane, kernel_size=1),
            nn.BatchNorm2d(pool_plane),
            nn.ReLU()
        )
    def forward(self,x):
        out1 = self.incep1(x)
        out2 = self.incep2(x)
        out3 = self.incep3(x)
        out4 = self.incep4(x)
        return torch.cat([out1,out2,out3,out4],dim=1)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3,192,kernel_size=3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.a3 = Inception(192,64,96,128,16,32,32)
        self.b3 = Inception(256,128,128,192,32,96,64)

        self.max_pool = nn.MaxPool2d(3,stride=2,padding=1)

        self.a4 = Inception(480,192,96,208,16,48,64)
        self.b4 = Inception(512,160,112,224,24,64,64)
        self.c4 = Inception(512,128,128,256,24,64,64)
        self.d4 = Inception(512,112,144,288,32,64,64)
        self.e4 = Inception(528,256,160,320,32,128,128)

        self.a5 = Inception(832,256,160,320,32,128,128)
        self.b5 = Inception(832,384,192,384,48,128,128)

        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024,10)

    def forward(self,x):

        out = self.pre_conv(x)

        out = self.a3(out)
        out = self.b3(out)

        out = self.max_pool(out)

        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)

        out = self.max_pool(out)

        out = self.a5(out)
        out = self.b5(out)

        out = self.avg_pool(out)

        out = out.view(out.size(0),-1)

        out = self.linear(out)

        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = GoogleNet().to(device)

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
                path = 'cifar_GoogleNet/cifar10.pth'
                torch.save(cnn.state_dict(), path)
            print(loss_temp)
            running_loss = 0.0

print('finish!')

cnn=GoogleNet()
cnn.load_state_dict(torch.load('cifar_GoogleNet/cifar10.pth'))
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
