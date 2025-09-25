'''VGG11/13/16/19 in Pytorch.

Reference:
[1] Karen Simonyan and Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556

Note: 
- To keep this VGG implementation consistent with the ResNet implementation, the "with_latent" argument to the forward method is renamed to "return_features", and the forward_all_features method is added.
- nn.AvgPool2d is replaced with nn.AdaptiveAvgPool2d to make this VGG implementation also work with larger image sizes.
- The torchsummary import is replaced by torchinfo, as the module has been renamed.
'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        out = self.features(x)
        latent = out.view(out.size(0), -1)
        out = self.classifier(latent)
        if return_features:
            return out, latent
        return out
    
    # Return features after every MaxPool2d layer
    def forward_all_features(self, x):
        n_layers = len(self.features)
        maxpool_indices = [i for i in range(n_layers) if isinstance(self.features[i], nn.MaxPool2d)]
        features = []
        start = 0

        for end in maxpool_indices:
            x = self.features[start:end+1](x)
            features.append(x)
            start = end+1

        x = self.features[start:](x)
        latent = x.view(x.size(0), -1)
        out = self.classifier(latent)

        return out, features

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d(output_size=(1,1))]
        return nn.Sequential(*layers)

def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13', **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)


if __name__ == "__main__":
    from torchinfo import summary
    model = VGG16(num_classes=10)
    summary(model, (1, 3, 32, 32), device='cpu')