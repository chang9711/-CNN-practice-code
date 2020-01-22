import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from torch.autograd import Variable

batch_size = 100
epoch_num = 5
learning_rate = 0.001

train_set = dsets.MNIST(root='./Desktop', train=True, download=True, transform=transforms.ToTensor())
test_set = dsets.MNIST(root='./Desktop', train=False, download=True, transform=transforms.ToTensor())

train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*7*7,10)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


cnn=CNN()

if os.path.isfile('pkl2/cnn.pkl'):
    cnn.load_state_dict(torch.load('pkl2/cnn.pkl'))
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)

    for epoch in range(epoch_num):
        for i,(image,label) in enumerate(train_set_loader):
            image = Variable(image)
            label = Variable(label)

            optimizer.zero_grad()
            output=cnn(image)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if (i+1)%100==0:
                print('Epoch [%d/%d], iter [%d/%d] Loss: %.4f'%(epoch+1,epoch_num,i+1,len(train_set)//batch_size,loss.data))
                if not os.path.isfile('pkl2/cnn.pkl'):
                    torch.save(cnn.state_dict(),'pkl2/cnn.pkl')

cnn.eval()
correct = 0
total = 0

for image,label in test_set_loader:
    image = Variable(image)
    output = cnn(image)
    _,predict = torch.max(output.data,1)
    total+=label.size(0)
    correct+=(predict == label).sum()


print('test Accuracy 10000 test images =%f %%'%(100*correct/total))











