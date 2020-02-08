import itertools
import math

import torch


class PriorBox(object):
    """
    Author: Max deGroot : https://github.com/amdegroot/ssd.pytorch
    from https://github.com/amdegroot/ssd.pytorch ssd.pytorch/layers/functions/prior_box.py file
    """

    def __init__(self, image_size, aspect_ratios, feature_maps, steps, clip, minimum_sizes, maximum_sizes):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.minimum_sizes = minimum_sizes
        self.maximum_sizes = maximum_sizes
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.minimum_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.maximum_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
