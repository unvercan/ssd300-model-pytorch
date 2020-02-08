"""
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot
from https://github.com/amdegroot/ssd.pytorch ssd.pytorch/data/voc0712.py file
"""
configuration = {
    'labels': [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ],
    'channel_means': (104, 117, 123),
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'image_size': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'minimum_sizes': [30, 60, 111, 162, 213, 264],
    'maximum_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variances': [0.1, 0.2],
    'clip': True,
    'input_channel': 3,
    'offsets': 4,
    'gamma': 20,
    'nms_threshold': 0.45,
    'confidence_threshold': 0.01,
    'top': 200
}
