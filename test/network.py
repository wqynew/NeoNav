import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import subprocess
import pdb



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

device = "cuda" if torch.cuda.is_available() else 'cpu'

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0c= torch.nn.Linear(2048, 1024)
        self.fc0c_1= torch.nn.Linear(2048, 1024)
        self.fca0c=torch.nn.Linear(1024*7, 1024*6)
        self.fca1c=torch.nn.Linear(1024*6, 1024*5)
        self.fca2c=torch.nn.Linear(1024*5, 1024*4)
        self.fcbc=torch.nn.Linear(1024*4, 1024*2)
        self.fc1c = torch.nn.Linear(2048, 1024)
        self.fc2c_mean = torch.nn.Linear(1024, 400)
        self.fc2c_sigma = torch.nn.Linear(1024, 400)
        self.tfc2c = torch.nn.Linear(400, 1024)
        self.tfc1c = torch.nn.Linear(1024, 2048)
        self.fprior_a= torch.nn.Linear(7, 1024)
        self.fprior_xa=torch.nn.Linear(1024*2,1024)
        self.fprior_mean = torch.nn.Linear(1024, 400)
        self.fprior_sigma = torch.nn.Linear(1024, 400)
        self.fa_pre= torch.nn.Linear(7, 1024)
        self.facom = torch.nn.Linear(5120, 2048)
        self.fa1 = torch.nn.Linear(2048, 1024)
        self.fa2= torch.nn.Linear(1024, 512)
        self.fa3= torch.nn.Linear(512, 7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.01)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x=x.transpose(1,3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_prior(self,input_image,actions):
        softplus = nn.Softplus()
        im_feat0=self.forward_once(input_image)
        fprior_a=F.relu(self.fprior_a(actions))
        fprior_x1=F.relu(self.fc0c(im_feat0))
        fprior_xa=torch.cat((fprior_x1,fprior_a),1)
        fprior_xa=F.relu(self.fprior_xa(fprior_xa))
        fprior_mean=self.fprior_mean(fprior_xa)
        fprior_sigma=softplus(self.fprior_sigma(fprior_xa))
        return fprior_mean,fprior_sigma

    def forward(self, x0,x1,x2,x3,x4, input_image,goal_image, actions, preact, batch_size):
        self.batch_size=batch_size
        im_feat0=self.forward_once(input_image)
        im_feat1=F.relu(self.fc0c_1(im_feat0))
        pref0=self.forward_once(x0)
        pref0=F.relu(self.fc0c_1(pref0))
        pref1=self.forward_once(x1)
        pref1=F.relu(self.fc0c_1(pref1))
        pref2=self.forward_once(x2)
        pref2=F.relu(self.fc0c_1(pref2))
        pref3=self.forward_once(x3)
        pref3=F.relu(self.fc0c_1(pref3))
        pref4=self.forward_once(x4)
        pref4=F.relu(self.fc0c_1(pref4))
        im_feat2=self.forward_once(goal_image)
        im_feat2=F.relu(self.fc0c_1(im_feat2))
        im_feat1=torch.cat((im_feat2, pref0, pref1, pref2, pref3, pref4, im_feat1),1)
        fcac=F.relu(self.fca0c(im_feat1))
        fcac=F.relu(self.fca1c(fcac))
        fcac=F.relu(self.fca2c(fcac))
        fcbc=F.relu(self.fcbc(fcac))
        fc1c=F.relu(self.fc1c(fcbc))
        z_mean=self.fc2c_mean(fc1c)
        softplus = nn.Softplus()
        z_sigma=softplus(self.fc2c_sigma(fc1c))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(self.batch_size, 400, device=torch.device("cuda"))
        tfc2c=F.relu(self.tfc2c(z))
        im_xtilt=F.relu(self.tfc1c(tfc2c))
        fa_pre=F.relu(self.fa_pre(preact))
        premid_im_feat=torch.cat((im_feat0,im_xtilt,fa_pre),1)
        facom=F.relu(self.facom(premid_im_feat))
        fa1=F.relu(self.fa1(facom))
        fa2=F.relu(self.fa2(fa1))
        action_prob=self.fa3(fa2)
        return z_mean,z_sigma, im_xtilt, action_prob

    def chooseAction(self, x0,x1,x2,x3,x4,current,goal,preact):   
        pref0=self.forward_once(x0)
        pref0=F.relu(self.fc0c_1(pref0))
        pref1=self.forward_once(x1)
        pref1=F.relu(self.fc0c_1(pref1))
        pref2=self.forward_once(x2)
        pref2=F.relu(self.fc0c_1(pref2))
        pref3=self.forward_once(x3)
        pref3=F.relu(self.fc0c_1(pref3))  
        pref4=self.forward_once(x4)
        pref4=F.relu(self.fc0c_1(pref4))
        im_feat0=self.forward_once(current)
        im_feat1=F.relu(self.fc0c_1(im_feat0))
        im_feat2=self.forward_once(goal)
        im_feat2=F.relu(self.fc0c_1(im_feat2))
        im_feat1=torch.cat((im_feat2, pref0, pref1, pref2, pref3, pref4, im_feat1),1)
        fcac=F.relu(self.fca0c(im_feat1))
        fcac=F.relu(self.fca1c(fcac))
        fcac=F.relu(self.fca2c(fcac))
        fcbc=F.relu(self.fcbc(fcac))
        fc1c=F.relu(self.fc1c(fcbc))
        z_mean=self.fc2c_mean(fc1c)
        softplus = nn.Softplus()
        z_sigma=softplus(self.fc2c_sigma(fc1c))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(1, 400, device=torch.device("cuda"))
        tfc2c=F.relu(self.tfc2c(z))
        im_xtilt=F.relu(self.tfc1c(tfc2c))
        fa_pre=F.relu(self.fa_pre(preact))
        premid_im_feat=torch.cat((im_feat0,im_xtilt,fa_pre),1)
        facom=F.relu(self.facom(premid_im_feat))
        fa1=F.relu(self.fa1(facom))
        fa2=F.relu(self.fa2(fa1))
        action_prob=self.fa3(fa2)
        return action_prob

    def chooseActionx(self, x0,x1,x2,x3,x4,current,goal,preact):   
        self.batch_size=10 
        pref0=F.relu(self.fc0c_1(x0))
        pref1=F.relu(self.fc0c_1(x1))
        pref2=F.relu(self.fc0c_1(x2))
        pref3=F.relu(self.fc0c_1(x3))
        pref4=F.relu(self.fc0c_1(x4))
        im_feat1=F.relu(self.fc0c_1(current))
        im_feat2=F.relu(self.fc0c_1(goal))
        im_feat1=torch.cat((im_feat2, pref0, pref1, pref2, pref3, pref4, im_feat1),1)
        fcac=F.relu(self.fca0c(im_feat1))
        fcac=F.relu(self.fca1c(fcac))
        fcac=F.relu(self.fca2c(fcac))
        fcbc=F.relu(self.fcbc(fcac))
        fc1c=F.relu(self.fc1c(fcbc))
        z_mean=self.fc2c_mean(fc1c)
        softplus = nn.Softplus()
        z_sigma=softplus(self.fc2c_sigma(fc1c))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(1, 400)
        tfc2c=F.relu(self.tfc2c(z))
        im_xtilt=F.relu(self.tfc1c(tfc2c))
        fa_pre=F.relu(self.fa_pre(preact))
        premid_im_feat=torch.cat((current,im_xtilt,fa_pre),1)
        facom=F.relu(self.facom(premid_im_feat))
        fa1=F.relu(self.fa1(facom))
        fa2=F.relu(self.fa2(fa1))
        action_prob=self.fa3(fa2)
        return action_prob

    def initial_para(self):
        torch.nn.init.xavier_uniform_(self.fa0.weight.data)
        self.fa0.bias.data.fill_(0.01)

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)

def create_model(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.apply(weights_init)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)

    return model
    if ema:
        for param in model.parameters():
            param.detach_()

    return model