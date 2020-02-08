import os
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable

from ssd300.detection import Detection
from ssd300.l2_normalization import L2Normalization
from ssd300.prior_box import PriorBox


# feature layers based on vgg16 feature layers
def generate_vgg(input_channel=3):
    return OrderedDict([
        ('conv_1_1', nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1)),
        ('relu_1_1', nn.ReLU(inplace=True)),
        ('conv_1_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
        ('relu_1_2', nn.ReLU(inplace=True)),
        ('pool_1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ('conv_2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
        ('relu_2_1', nn.ReLU(inplace=True)),
        ('conv_2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
        ('relu_2_2', nn.ReLU(inplace=True)),
        ('pool_2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ('conv_3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
        ('relu_3_1', nn.ReLU(inplace=True)),
        ('conv_3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
        ('relu_3_2', nn.ReLU(inplace=True)),
        ('conv_3_3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
        ('relu_3_3', nn.ReLU(inplace=True)),
        ('pool_3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)),
        ('conv_4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_4_1', nn.ReLU(inplace=True)),
        ('conv_4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_4_2', nn.ReLU(inplace=True)),
        ('conv_4_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_4_3', nn.ReLU(inplace=True)),
        ('pool_4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ('conv_5_1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_5_1', nn.ReLU(inplace=True)),
        ('conv_5_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_5_2', nn.ReLU(inplace=True)),
        ('conv_5_3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
        ('relu_5_3', nn.ReLU(inplace=True)),
        ('pool_5', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
        ('conv_6', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=6, dilation=6)),
        ('relu_6', nn.ReLU(inplace=True)),
        ('conv_7', nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)),
        ('relu_7', nn.ReLU(inplace=True))
    ])


# feature scaling layers
def generate_scaling():
    return OrderedDict([
        ('conv_8_1', nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)),
        ('relu_8_1', nn.ReLU(inplace=True)),
        ('conv_8_2', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)),
        ('relu_8_2', nn.ReLU(inplace=True)),
        ('conv_9_1', nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)),
        ('relu_9_1', nn.ReLU(inplace=True)),
        ('conv_9_2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)),
        ('relu_9_2', nn.ReLU(inplace=True)),
        ('conv_10_1', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)),
        ('relu_10_1', nn.ReLU(inplace=True)),
        ('conv_10_2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)),
        ('relu_10_2', nn.ReLU(inplace=True)),
        ('conv_11_1', nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)),
        ('relu_11_1', nn.ReLU(inplace=True)),
        ('conv_11_2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)),
        ('relu_11_2', nn.ReLU(inplace=True))
    ])


# prior boxes
def generate_prior_boxes():
    return {
        'conv_4_3_norm': 4,
        'conv_7': 6,
        'conv_8_2': 6,
        'conv_9_2': 6,
        'conv_10_2': 4,
        'conv_11_2': 4
    }


# confidence score layers
def generate_confidence(vgg, scaling, prior_boxes, classes=20):
    # output channel = prior_box * (classes + background) for each location
    return OrderedDict([
        ('conv_4_3_norm', nn.Conv2d(in_channels=vgg['conv_4_3'].out_channels, kernel_size=3, stride=1, padding=1,
                                    out_channels=(prior_boxes['conv_4_3_norm'] * (classes + 1)))),
        ('conv_7', nn.Conv2d(in_channels=vgg['conv_7'].out_channels, kernel_size=3, stride=1, padding=1,
                             out_channels=(prior_boxes['conv_7'] * (classes + 1)))),
        ('conv_8_2', nn.Conv2d(in_channels=scaling['conv_8_2'].out_channels, kernel_size=3, stride=1, padding=1,
                               out_channels=(prior_boxes['conv_8_2'] * (classes + 1)))),
        ('conv_9_2', nn.Conv2d(in_channels=scaling['conv_9_2'].out_channels, kernel_size=3, stride=1, padding=1,
                               out_channels=(prior_boxes['conv_9_2'] * (classes + 1)))),
        ('conv_10_2', nn.Conv2d(in_channels=scaling['conv_10_2'].out_channels, kernel_size=3, stride=1, padding=1,
                                out_channels=(prior_boxes['conv_10_2'] * (classes + 1)))),
        ('conv_11_2', nn.Conv2d(in_channels=scaling['conv_11_2'].out_channels, kernel_size=3, stride=1, padding=1,
                                out_channels=(prior_boxes['conv_11_2'] * (classes + 1))))
    ])


# location offset layers
def generate_location(vgg, scaling, prior_boxes, offsets=4):
    # output channel = prior_box * (center_x, center_y, height, width) for each location
    return OrderedDict([
        ('conv_4_3_norm', nn.Conv2d(in_channels=vgg['conv_4_3'].out_channels, kernel_size=3, stride=1, padding=1,
                                    out_channels=(prior_boxes['conv_4_3_norm'] * offsets))),
        ('conv_7', nn.Conv2d(in_channels=vgg['conv_7'].out_channels, kernel_size=3, stride=1, padding=1,
                             out_channels=(prior_boxes['conv_7'] * offsets))),
        ('conv_8_2', nn.Conv2d(in_channels=scaling['conv_8_2'].out_channels, kernel_size=3, stride=1, padding=1,
                               out_channels=(prior_boxes['conv_8_2'] * offsets))),
        ('conv_9_2', nn.Conv2d(in_channels=scaling['conv_9_2'].out_channels, kernel_size=3, stride=1, padding=1,
                               out_channels=(prior_boxes['conv_9_2'] * offsets), )),
        ('conv_10_2', nn.Conv2d(in_channels=scaling['conv_10_2'].out_channels, kernel_size=3, stride=1, padding=1,
                                out_channels=(prior_boxes['conv_10_2'] * offsets))),
        ('conv_11_2', nn.Conv2d(in_channels=scaling['conv_11_2'].out_channels, kernel_size=3, stride=1, padding=1,
                                out_channels=(prior_boxes['conv_11_2'] * offsets)))
    ])


class SSD300(nn.Module):
    def __init__(self, base, scaling, location, confidence, classes, configuration):
        super(SSD300, self).__init__()
        self.classes = classes
        self.configuration = configuration
        self.prior_box = PriorBox(image_size=configuration['image_size'], aspect_ratios=configuration['aspect_ratios'],
                                  feature_maps=configuration['feature_maps'], steps=configuration['steps'],
                                  minimum_sizes=configuration['minimum_sizes'], clip=configuration['clip'],
                                  maximum_sizes=configuration['maximum_sizes'])
        self.priors = Variable(self.prior_box.forward(), volatile=True)
        self.base = nn.ModuleDict(modules=base)
        self.l2_normalization = L2Normalization(input_channel=base['conv_4_3'].out_channels,
                                                gamma=configuration['gamma'])
        self.scaling = nn.ModuleDict(modules=scaling)
        self.location = nn.ModuleDict(modules=location)
        self.confidence = nn.ModuleDict(modules=confidence)
        self.soft_max = nn.Softmax(dim=-1)
        self.detector = Detection(classes=classes, variances=configuration['variances'], top=configuration['top'],
                                  confidence_threshold=configuration['confidence_threshold'],
                                  nms_threshold=configuration['nms_threshold'])

    def forward(self, x):
        # batch size
        batch_size = x.size(0)

        # multibox
        locations = list()
        confidences = list()

        # base feature layers forward pass
        for key in self.base.keys():
            x = self.base[key](x)
            if key == 'relu_4_3':
                conv_4_3_norm = self.l2_normalization(x)
                locations.append(self.location['conv_4_3_norm'](conv_4_3_norm))
                confidences.append(self.confidence['conv_4_3_norm'](conv_4_3_norm))
            elif key == 'relu_7':
                locations.append(self.location['conv_7'](x))
                confidences.append(self.confidence['conv_7'](x))

        # feature scaling layers forward pass
        for key in self.scaling.keys():
            x = self.scaling[key](x)
            if key == 'relu_8_2':
                locations.append(self.location['conv_8_2'](x))
                confidences.append(self.confidence['conv_8_2'](x))
            elif key == 'relu_9_2':
                locations.append(self.location['conv_9_2'](x))
                confidences.append(self.confidence['conv_9_2'](x))
            elif key == 'relu_10_2':
                locations.append(self.location['conv_10_2'](x))
                confidences.append(self.confidence['conv_10_2'](x))
            elif key == 'relu_11_2':
                locations.append(self.location['conv_11_2'](x))
                confidences.append(self.confidence['conv_11_2'](x))

        # reorder axises:
        # (batch, channel, height, width) -> (batch, height, width, channel)
        # flatten by batch:
        # (batch, height, width, channel) -> (batch, (height x width x channel))
        for index, tensor in enumerate(locations):
            reordered = tensor.permute(0, 2, 3, 1).contiguous()
            reordered_flatten_by_batch = reordered.view(batch_size, -1)
            locations[index] = reordered_flatten_by_batch
        for index, tensor in enumerate(confidences):
            reordered = tensor.permute(0, 2, 3, 1).contiguous()
            reordered_flatten_by_batch = reordered.view(batch_size, -1)
            confidences[index] = reordered_flatten_by_batch

        # concatenate by layers:
        # layer x (batch, (height x width x channel)) -> (batch, (layer x height x width x channel))
        location_tensor = torch.cat(tensors=locations, dim=1)
        confidence_tensor = torch.cat(tensors=confidences, dim=1)

        return self.detector(
            location_tensor.view(batch_size, -1, 4),
            self.soft_max(confidence_tensor.view(batch_size, -1, self.classes)),
            self.priors.type(type(x.data))
        )

    def custom_pre_trained_loader(self, pre_trained_file):
        _, extension = os.path.splitext(pre_trained_file)
        supported_extensions = ['.pkl', '.pth']
        if extension in supported_extensions:
            pre_trained = torch.load(f=pre_trained_file, map_location=lambda storage, loc: storage)
            print('Loading weights into state dict...')

            self.base['conv_1_1'].bias.data.copy_(other=pre_trained['vgg.0.bias'].data)
            self.base['conv_1_1'].weight.data.copy_(other=pre_trained['vgg.0.weight'].data)

            self.base['conv_1_2'].bias.data.copy_(other=pre_trained['vgg.2.bias'].data)
            self.base['conv_1_2'].weight.data.copy_(other=pre_trained['vgg.2.weight'].data)

            self.base['conv_2_1'].bias.data.copy_(other=pre_trained['vgg.5.bias'].data)
            self.base['conv_2_1'].weight.data.copy_(other=pre_trained['vgg.5.weight'].data)

            self.base['conv_2_2'].bias.data.copy_(other=pre_trained['vgg.7.bias'].data)
            self.base['conv_2_2'].weight.data.copy_(other=pre_trained['vgg.7.weight'].data)

            self.base['conv_3_1'].bias.data.copy_(other=pre_trained['vgg.10.bias'].data)
            self.base['conv_3_1'].weight.data.copy_(other=pre_trained['vgg.10.weight'].data)

            self.base['conv_3_2'].bias.data.copy_(other=pre_trained['vgg.12.bias'].data)
            self.base['conv_3_2'].weight.data.copy_(other=pre_trained['vgg.12.weight'].data)

            self.base['conv_3_3'].bias.data.copy_(other=pre_trained['vgg.14.bias'].data)
            self.base['conv_3_3'].weight.data.copy_(other=pre_trained['vgg.14.weight'].data)

            self.base['conv_4_1'].bias.data.copy_(other=pre_trained['vgg.17.bias'].data)
            self.base['conv_4_1'].weight.data.copy_(other=pre_trained['vgg.17.weight'].data)

            self.base['conv_4_2'].bias.data.copy_(other=pre_trained['vgg.19.bias'].data)
            self.base['conv_4_2'].weight.data.copy_(other=pre_trained['vgg.19.weight'].data)

            self.base['conv_4_3'].bias.data.copy_(other=pre_trained['vgg.21.bias'].data)
            self.base['conv_4_3'].weight.data.copy_(other=pre_trained['vgg.21.weight'].data)

            self.base['conv_5_1'].bias.data.copy_(other=pre_trained['vgg.24.bias'].data)
            self.base['conv_5_1'].weight.data.copy_(other=pre_trained['vgg.24.weight'].data)

            self.base['conv_5_2'].bias.data.copy_(other=pre_trained['vgg.26.bias'].data)
            self.base['conv_5_2'].weight.data.copy_(other=pre_trained['vgg.26.weight'].data)

            self.base['conv_5_3'].bias.data.copy_(other=pre_trained['vgg.28.bias'].data)
            self.base['conv_5_3'].weight.data.copy_(other=pre_trained['vgg.28.weight'].data)

            self.base['conv_6'].bias.data.copy_(other=pre_trained['vgg.31.bias'].data)
            self.base['conv_6'].weight.data.copy_(other=pre_trained['vgg.31.weight'].data)

            self.base['conv_7'].bias.data.copy_(other=pre_trained['vgg.33.bias'].data)
            self.base['conv_7'].weight.data.copy_(other=pre_trained['vgg.33.weight'].data)

            self.l2_normalization.weight.data.copy_(other=pre_trained['L2Norm.weight'].data)

            self.scaling['conv_8_1'].bias.data.copy_(other=pre_trained['extras.0.bias'].data)
            self.scaling['conv_8_1'].weight.data.copy_(other=pre_trained['extras.0.weight'].data)

            self.scaling['conv_8_2'].bias.data.copy_(other=pre_trained['extras.1.bias'].data)
            self.scaling['conv_8_2'].weight.data.copy_(other=pre_trained['extras.1.weight'].data)

            self.scaling['conv_9_1'].bias.data.copy_(other=pre_trained['extras.2.bias'].data)
            self.scaling['conv_9_1'].weight.data.copy_(other=pre_trained['extras.2.weight'].data)

            self.scaling['conv_9_2'].bias.data.copy_(other=pre_trained['extras.3.bias'].data)
            self.scaling['conv_9_2'].weight.data.copy_(other=pre_trained['extras.3.weight'].data)

            self.scaling['conv_10_1'].bias.data.copy_(other=pre_trained['extras.4.bias'].data)
            self.scaling['conv_10_1'].weight.data.copy_(other=pre_trained['extras.4.weight'].data)

            self.scaling['conv_10_2'].bias.data.copy_(other=pre_trained['extras.5.bias'].data)
            self.scaling['conv_10_2'].weight.data.copy_(other=pre_trained['extras.5.weight'].data)

            self.scaling['conv_11_1'].bias.data.copy_(other=pre_trained['extras.6.bias'].data)
            self.scaling['conv_11_1'].weight.data.copy_(other=pre_trained['extras.6.weight'].data)

            self.scaling['conv_11_2'].bias.data.copy_(other=pre_trained['extras.7.bias'].data)
            self.scaling['conv_11_2'].weight.data.copy_(other=pre_trained['extras.7.weight'].data)

            self.location['conv_4_3_norm'].bias.data.copy_(other=pre_trained['loc.0.bias'].data)
            self.location['conv_4_3_norm'].weight.data.copy_(other=pre_trained['loc.0.weight'].data)

            self.location['conv_7'].bias.data.copy_(other=pre_trained['loc.1.bias'].data)
            self.location['conv_7'].weight.data.copy_(other=pre_trained['loc.1.weight'].data)

            self.location['conv_8_2'].bias.data.copy_(other=pre_trained['loc.2.bias'].data)
            self.location['conv_8_2'].weight.data.copy_(other=pre_trained['loc.2.weight'].data)

            self.location['conv_9_2'].bias.data.copy_(other=pre_trained['loc.3.bias'].data)
            self.location['conv_9_2'].weight.data.copy_(other=pre_trained['loc.3.weight'].data)

            self.location['conv_10_2'].bias.data.copy_(other=pre_trained['loc.4.bias'].data)
            self.location['conv_10_2'].weight.data.copy_(other=pre_trained['loc.4.weight'].data)

            self.location['conv_11_2'].bias.data.copy_(other=pre_trained['loc.5.bias'].data)
            self.location['conv_11_2'].weight.data.copy_(other=pre_trained['loc.5.weight'].data)

            self.confidence['conv_4_3_norm'].bias.data.copy_(other=pre_trained['conf.0.bias'].data)
            self.confidence['conv_4_3_norm'].weight.data.copy_(other=pre_trained['conf.0.weight'].data)

            self.confidence['conv_7'].bias.data.copy_(other=pre_trained['conf.1.bias'].data)
            self.confidence['conv_7'].weight.data.copy_(other=pre_trained['conf.1.weight'].data)

            self.confidence['conv_8_2'].bias.data.copy_(other=pre_trained['conf.2.bias'].data)
            self.confidence['conv_8_2'].weight.data.copy_(other=pre_trained['conf.2.weight'].data)

            self.confidence['conv_9_2'].bias.data.copy_(other=pre_trained['conf.3.bias'].data)
            self.confidence['conv_9_2'].weight.data.copy_(other=pre_trained['conf.3.weight'].data)

            self.confidence['conv_10_2'].bias.data.copy_(other=pre_trained['conf.4.bias'].data)
            self.confidence['conv_10_2'].weight.data.copy_(other=pre_trained['conf.4.weight'].data)

            self.confidence['conv_11_2'].bias.data.copy_(other=pre_trained['conf.5.bias'].data)
            self.confidence['conv_11_2'].weight.data.copy_(other=pre_trained['conf.5.weight'].data)

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_ssd300(configuration):
    prior_boxes = generate_prior_boxes()
    vgg = generate_vgg(input_channel=configuration['input_channel'])
    scaling = generate_scaling()
    location = generate_location(vgg=vgg, offsets=configuration['offsets'],
                                 prior_boxes=prior_boxes, scaling=scaling)
    confidence = generate_confidence(vgg=vgg, classes=len(configuration['labels']),
                                     scaling=scaling, prior_boxes=prior_boxes)
    ssd300 = SSD300(base=vgg, scaling=scaling, location=location,
                    classes=(len(configuration['labels']) + 1),
                    confidence=confidence, configuration=configuration)
    return ssd300
