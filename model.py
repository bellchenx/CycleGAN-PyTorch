import torch
import torch.nn as nn
import torch.nn.functional as F

class se_block_conv(nn.Module):
    def __init__(self, channel, kernel_size, stride, padding, enable):
        super(se_block_conv, self).__init__()
        self.enable = enable

        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        if self.enable:
            self.se_conv1 = nn.Conv2d(channel, channel//16, kernel_size=1)
            self.se_conv2 = nn.Conv2d(channel//16, channel, kernel_size=1)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        
        if self.enable:
            se = F.avg_pool2d(output, output.size(2))
            se = F.relu(self.se_conv1(se))
            se = F.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output

class se_block_deconv(nn.Module):
    def __init__(self, channel, kernel_size, stride, padding, enable):
        super(se_block_deconv, self).__init__()
        self.enable = enable

        self.conv1 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding, bias=True)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding, bias=True)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        self.se_conv1 = nn.Conv2d(channel, channel//16, kernel_size=1)
        self.se_conv2 = nn.Conv2d(channel//16, channel, kernel_size=1)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        
        if self.enable:
            se = F.avg_pool2d(output, output.size(2))
            se = F.relu(self.se_conv1(se))
            se = F.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        input_nc = config.G_input
        output_nc = config.G_output
        n_downsample = config.G_downsample
        ngf = config.G_channel
        nb = config.G_block
        enable_se = config.G_enable_se
        if config.G_block_type == 'deconv':
            block = [se_block_deconv(ngf * (2**n_downsample), kernel_size=3, stride=1, padding=1, enable=enable_se)]
        else:
            block = [se_block_conv(ngf * (2**n_downsample), kernel_size=3, stride=1, padding=1, enable=enable_se)]
        
        downsample = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
                        nn.InstanceNorm2d(ngf, affine=True),
                        nn.ReLU(True)]
        for i in range(n_downsample):
            mult = 2**i
            downsample += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                            nn.ReLU(True)]
        self.downsample = nn.Sequential(*downsample)

        resnet_blocks = []
        for i in range(nb):
            resnet_blocks += block
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        upsample = []
        for i in range(n_downsample):
            mult = 2**(n_downsample - i)
            upsample += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.InstanceNorm2d(ngf * mult // 2, affine=True),
                            nn.ReLU(True)]
        self.upsample = nn.Sequential(*upsample)
        self.final_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0)
        

    def forward(self, x):
        out = F.pad(x, (3, 3, 3, 3), 'reflect')
        out = self.downsample(out)
        out = self.resnet_blocks(out)
        out = self.upsample(out)
        out = F.pad(out, (3, 3, 3, 3), 'reflect')
        out = F.tanh(self.final_conv(out))
        return out

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        input_nc = config.D_input
        output_nc = config.D_output
        ndf = config.D_channel
        n_layers = config.D_downsample - 2

        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)]
        
        for i in range(n_layers):
            mult = 2**i
            stride=2
            if i >= 2:
                stride = 1
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=stride, padding=1),
                        nn.InstanceNorm2d(ndf * mult * 2, affine=True),
                        nn.LeakyReLU(0.2, True)]
        
        model += [nn.Conv2d(ndf * (2**n_layers), output_nc, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)