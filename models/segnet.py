import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights
from torchinfo import summary


class Encoder(nn.Module):
    """
    Encoder network consists first 13 conv layers of VGG16.
    See 
        print(vgg16_bn())
    """
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        # Conv and BN layers.
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(512)

        # MaxPooling and ReLU.
        # NOTE must return_index is True. see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        indices = []
        sizes = []       # see https://github.com/pytorch/pytorch/issues/580
        sizes.append(x.size())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x, index = self.pool(x)
        indices.append(index)
        sizes.append(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x, index = self.pool(x)
        indices.append(index)
        sizes.append(x.size())

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x, index = self.pool(x)
        indices.append(index)
        sizes.append(x.size())

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x, index = self.pool(x)
        indices.append(index)
        sizes.append(x.size())

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x, index = self.pool(x)
        indices.append(index)
        sizes.append(x.size())       

        return x, indices, sizes

class Decoder(nn.Module):

    def __init__(self, num_classes:int) -> None:
        super(Decoder, self).__init__()
        # Conv and BN layers
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, num_classes, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(num_classes)

        # Unpooling and ReLU
        # see https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, indices, sizes):
        x = self.unpool(x, indices[4], sizes[4])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.unpool(x, indices[3], sizes[3])
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.unpool(x, indices[2], sizes[2])
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)

        x = self.unpool(x, indices[1], sizes[1])
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)

        x = self.unpool(x, indices[0], sizes[0])
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        
        return x

class SegNet(nn.Module):

    def __init__(self, num_classes:int) -> None:
        super(SegNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        output = self.decoder(*self.encoder(x))
        return output


if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    segnet = SegNet(20)
    output = segnet(x)
    print(output.shape)