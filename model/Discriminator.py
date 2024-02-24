class Discriminator(nn.Module):
    def __init__(self,gpu):
        super(Generator,self).__init__()
        self.gpu = gpu
        #size of latent vector (z) = 100,ngf=64, 
        #input_size: 100, output=512, kernel size =4x4
        self.conv1 = nn.Conv2d(3,ngf,kernel_size=4, stride=2, padding=1,bias=False)
        self.bnorm1 = nn.BatchNorm2d(ngf)
        #input: 512, output=256, kernel size =4x4
        self.conv2 = nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1,bias=False)
        self.bnorm2 = nn.BatchNorm2d(ngf*2)
        #input: 256, output=128, kernel size =4x4
        self.conv3 = nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1,bias=False)
        self.bnorm3 = nn.BatchNorm2d(ngf*4)
        #input: 128, output=64, kernel size =4x4
        self.conv4 = nn.Conv2d(ngf*4,ngf*8,kernel_size=4,stride=2,padding=1,bias=False)
        self.bnorm4 = nn.BatchNorm2d(ngf*8)
        #input: 64, output=3, kernel size =4x4
        self.conv5 = nn.ConvTranspose2d(ngf*8,1,kernel_size=4,stride=2,padding=0,bias=False)
        
    def forward(self, input):
        x =  F.leakyrelu(self.bnorm1(self.conv1(input),0.2,True))
        x2 = F.leakyrelu(self.bnorm2(self.conv2(x),0.2,True))
        x3 = F.leakyrelu(self.bnorm3(self.conv2(x2),0.2,True))
        x4 = F.leakyrelu(self.bnorm4(self.conv2(x3),0.2,True))
        x5 = F.sigmoid(self.conv5(x4))
        return x5
