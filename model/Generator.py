
class Generator(nn.Module):
    def __init__(self, ngf):
        super(Generator, self).__init__()
        #size of latent vector (z) = 100,ngf=64, 
        #input_size: 100, output=512, kernel size =4x4
        self.convt1 = nn.ConvTranspose2d(100, ngf*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(ngf*8)
        #input: 512, output=256, kernel size =4x4
        self.conv2 = nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(ngf*4)
        #input: 256, output=128, kernel size =4x4
        self.conv3 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(ngf*2)
        #input: 128, output=64, kernel size =4x4
        self.conv4 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(ngf)
        #input: 64, output=3, kernel size =4x4
        self.conv5 = nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False)
        
    def forward(self, input):
        x = F.leaky_relu(self.bnorm1(self.convt1(input)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnorm2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnorm3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnorm4(self.conv4(x)), 0.2, inplace=True)
        x = torch.sigmoid(self.conv5(x))
        return x
