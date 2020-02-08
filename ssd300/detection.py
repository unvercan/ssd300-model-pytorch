import torch
from torch.autograd import Function


def decode(locations, priors, variances):
    """
    Adapted from https://github.com/Hakuyume/chainer-ssd by Max deGroot
    from https://github.com/amdegroot/ssd.pytorch ssd.pytorch/layers/box_utils.py file
    """
    boxes = torch.cat((
        priors[:, :2] + locations[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(locations[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top=200):
    """
    Original author: Francisco Massa: https://github.com/fmassa/object-detection.torch
    Ported to PyTorch by Max deGroot (02/01/2017)
    from https://github.com/amdegroot/ssd.pytorch ssd.pytorch/layers/box_utils.py file
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detection(Function):
    """
    Author: Max deGroot : https://github.com/amdegroot/ssd.pytorch
    from https://github.com/amdegroot/ssd.pytorch, ssd.pytorch/layers/functions/detection.py file
    """

    def __init__(self, classes, top, confidence_threshold, nms_threshold, variances):
        self.classes = classes
        self.top = top
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.variances = variances

    def forward(self, location_tensor, confidence_tensor, prior_tensor):
        num = location_tensor.size(0)  # batch size
        num_priors = prior_tensor.size(0)
        output = torch.zeros(num, self.classes, self.top, 5)
        conf_preds = confidence_tensor.view(num, num_priors, self.classes).transpose(2, 1)
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(location_tensor[i], prior_tensor, self.variances)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.classes):
                c_mask = conf_scores[cl].gt(self.confidence_threshold)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_threshold, self.top)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
