import paddle
import paddle.nn as nn
import numpy as np
def wasserstein_loss(y_true,y_pred):
    return paddle.mean(y_true*y_pred)
class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.counter=0
        self.loss_process=[]
        self.n_critic=5
        self.clip_value=0.01
    def _train(self,image,G):
        loss=[]
        for i in range(self.n_critic):
            z=paddle.randn([image.shape[0],100])
            fake_image=G(z)
            d_loss=wasserstein_loss(self.forward(image),-1)+wasserstein_loss(self.forward(fake_image.detach()),1)
            d_loss.backward()
            loss.append(d_loss.detach().numpy()[0])
            self.optim.step()
            self.optim.clear_grad()
            for p in self.parameters():
                p.clip(-self.clip_value,self.clip_value)
        if self.counter%10==0:
            self.loss_process.append(np.mean(loss))
        self.counter+=1
        return np.mean(loss)
    def setConfig(self,optim,loss_fun):
        self.optim=optim
        self.loss_fun=loss_fun
class DFC(Discriminator):
    def __init__(self):
        super(DFC, self).__init__()
        self.fc1=nn.Linear(784,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
    def forward(self,x):
        x=x.reshape([-1,784])
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class DConvNet(Discriminator):
    def __init__(self):
        super(DConvNet, self).__init__()
        self.conv1=nn.Conv2D(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2D(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=2)
        self.conv3=nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2)
        self.bn1=nn.BatchNorm2D(32)
        self.bn2=nn.BatchNorm2D(64)
        self.bn3=nn.BatchNorm2D(128)
        self.pool1=nn.AvgPool2D(kernel_size=2,stride=2)
        self.pool2=nn.AvgPool2D(kernel_size=2,stride=2)
        self.relu=nn.ReLU()
        self.avgpool=nn.AdaptiveAvgPool2D(1)
        self.fc1=nn.Linear(128,32)
        self.fc2=nn.Linear(32,1)
        self.sigmoid=nn.Sigmoid()
        self.loss_process=[]
        self.counter=0
    def forward(self,x):
        x=self.pool1(self.relu(self.conv1(x)))
        x=self.bn1(x)
        x=self.pool2(self.relu(self.conv2(x)))
        x=self.bn2(x)
        x=self.relu(self.conv3(x))
        x=self.bn3(x)
        x=self.avgpool(x)
        x=x.flatten(1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
if __name__=='__main__':
    model=DFC()
    A=[1,2,3,4]
    print(np.mean(A))

