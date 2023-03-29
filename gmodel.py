import paddle
import paddle.nn as nn
def wasserstein_loss(y_true,y_pred):
    return paddle.mean(y_true*y_pred)
class GConvNet(nn.Layer):
    def __init__(self,in_dim):
        super(GConvNet, self).__init__()
        self.proj=nn.Linear(in_dim,256*7*7)
        self.tconv1=nn.Conv2DTranspose(in_channels=256,out_channels=128,stride=2,kernel_size=3,padding=1,output_padding=1) # 8
        self.tconv2=nn.Conv2DTranspose(in_channels=128,out_channels=64,stride=2,kernel_size=3,padding=1,output_padding=1) # 16
        self.tconv3=nn.Conv2DTranspose(in_channels=64,out_channels=32,stride=1,kernel_size=3,padding=1) # 20
        self.tconv4=nn.Conv2DTranspose(in_channels=32,out_channels=16,stride=1,kernel_size=3,padding=1) # 20
        self.tconv5=nn.Conv2DTranspose(in_channels=16,out_channels=1,stride=1,kernel_size=3,padding=1) # 20
        self.bn=nn.BatchNorm2D(256)
        self.bn1=nn.BatchNorm2D(128)
        self.bn2=nn.BatchNorm2D(64)
        self.bn3=nn.BatchNorm2D(32)
        self.bn4=nn.BatchNorm2D(16)
        self.tanh=nn.Sigmoid()
        self.relu=nn.LeakyReLU(0.2)
        self.counter=0
        self.loss_process=[]
    def setConfig(self,optim):
        self.optim=optim
    def forward(self,x):
        x=self.proj(x)
        x=x.reshape([-1,256,7,7])
        x=self.relu(self.bn(x))
        x=self.relu(self.bn1(self.tconv1(x)))
        x=self.relu(self.bn2(self.tconv2(x)))
        x=self.relu(self.bn3(self.tconv3(x)))
        x=self.relu(self.bn4(self.tconv4(x)))
        x=self.tanh(self.tconv5(x))
        return x
    def _train(self,D,image):
        z=paddle.randn([image.shape[0],100])
        fake_image=self.forward(z)
        g_loss=-paddle.mean(D(fake_image))
        g_loss.backward()
        self.optim.step()
        self.optim.clear_grad()
        if self.counter%10==0:
            self.loss_process.append(g_loss.detach().numpy()[0])
        self.counter+=1
        return g_loss.detach().numpy()[0]
if __name__=='__main__':
    t=paddle.ones([1,100])
    model=GConvNet(100)
    print(model(t).shape)
