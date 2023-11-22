import argparse

from torch.autograd import Variable

from models.FasterNet import load_backbone
import torch
import torch.nn.functional as F
from torch import nn
from models.yolo_pafpn1 import YOLOPAFPN
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from models.CBAM import CBAMBlock
from models.matcher import build_matcher_crowd
from models.afpn import AFPN
import numpy as np
from models.MSFM import MSFM
import time

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1) #B C H W => B H W C

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256,C2_size=64):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5) #Change Channel
        P5_upsampled_x = self.P5_upsampled(P5_x) # Up-sampling
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        # P3_upsampled_x = self.P4_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        # P2_x = self.P2_1(C3)
        # P2_x = P2_x + P3_upsampled_x
        # P2_x = self.P2_2(P2_x)

        return [P3_x, P4_x, P5_x]


# the defenition of the M2PLNet model
class M2PLNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line
        self.channel_nums=768
        self.regression0 = RegressionModel(num_features_in=self.channel_nums, num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.classification0 = ClassificationModel(num_features_in=self.channel_nums, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points, feature_size=self.channel_nums)


        self.regression = RegressionModel(num_features_in=self.channel_nums, num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.classification = ClassificationModel(num_features_in=self.channel_nums, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.regression1 = RegressionModel(num_features_in=self.channel_nums, num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.classification1 = ClassificationModel(num_features_in=self.channel_nums, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.regression2 = RegressionModel(num_features_in=self.channel_nums, num_anchor_points=num_anchor_points, feature_size=self.channel_nums)
        self.classification2 = ClassificationModel(num_features_in=self.channel_nums, \
                                                   num_classes=self.num_classes, \
                                                   num_anchor_points=num_anchor_points, feature_size=self.channel_nums)

        self.anchor_points = AnchorPoints(pyramid_levels=[2,3,4,5], row=row, line=line)
        self.cfp=AFPN(in_channels=[192,384, 768, 1536],out_channels=self.channel_nums)
        self.cbam=CBAMBlock(channel=self.channel_nums,kernel_size=3)
        self.cbam0=CBAMBlock(channel=self.channel_nums,kernel_size=3)
        self.cbam1=CBAMBlock(channel=self.channel_nums,kernel_size=3)
        self.cbam2=CBAMBlock(channel=self.channel_nums,kernel_size=3)
        self.msfm=MSFM(1536,1536)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features[3]=self.msfm(features[3])
        features_cfp=self.cfp([features[0],features[1], features[2], features[3]])
        # run the regression and classification branch
        features_cfp[0]=self.cbam0(features_cfp[0])
        features_cfp[1]=self.cbam(features_cfp[1])
        features_cfp[2]=self.cbam1(features_cfp[2])
        features_cfp[3]=self.cbam2(features_cfp[3])
        regression0 = self.regression0(features_cfp[0]) * 100  # 8x
        classification0 = self.classification0(features_cfp[0])

        regression = self.regression(features_cfp[1]) * 100 # 8x
        classification = self.classification(features_cfp[1])

        regression1 = self.regression1(features_cfp[2]) * 100  # 16x
        classification1 = self.classification1(features_cfp[2])

        regression2 = self.regression2(features_cfp[3]) * 100  # 32x
        classification2 = self.classification2(features_cfp[3])


        regression=torch.cat([regression0,regression, regression1,regression2], dim=1)
        classification=torch.cat([classification0,classification, classification1,classification2], dim=1)
        anchor_points = self.anchor_points(samples)

        # decode the points as prediction
        output_coord = regression.cuda() + anchor_points

        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}


        return out





class Focal_L1(nn.Module):
    def __init__(self):
        super(Focal_L1, self).__init__()
        self.epsilon=torch.tensor(0.5)
    def forward(self, x, y):
        if isinstance(x, (float, int)): x = Variable(torch.Tensor([x]))
        if isinstance(y, (float, int)): y = Variable(torch.Tensor([y]))
        # x=Variable(x)
        # (predict_cnt - gt_cnt) * abs(torch.log2(abs(predict_cnt - gt_cnt) / (gt_cnt + 1e-6)))
        m=torch.abs(torch.mul((x-y),(torch.sqrt(torch.abs(x-y)/(y+self.epsilon)))))
        mse_loss = torch.mean(torch.abs(torch.mul((x-y),(torch.sqrt(torch.abs(x-y)/(y+self.epsilon))))))
        return mse_loss

class Focal_L2(nn.Module):
    def __init__(self):
        super(Focal_L2, self).__init__()
        self.epsilon=torch.tensor(0.5)
    def forward(self, x, y):
        if isinstance(x, (float, int)): x = Variable(torch.Tensor([x]))
        if isinstance(y, (float, int)): y = Variable(torch.Tensor([y]))
        # x=Variable(x)
        # (predict_cnt - gt_cnt) * abs(torch.log2(abs(predict_cnt - gt_cnt) / (gt_cnt + 1e-6)))
        # m=torch.abs(torch.mul((x-y),(torch.sqrt(torch.abs(x-y)/(y+self.epsilon)))))
        mse_loss = torch.mean(torch.abs(torch.mul((x-y)/(y+self.epsilon),(torch.log2(torch.abs(x-y)+torch.tensor(1))))))
        return mse_loss


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = 0.25
        empty_weight[1] = (1-self.eos_coef)
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # print(self.empty_weight)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
        loss_bbox = torch.log2(F.smooth_l1_loss(src_points, target_points, reduction='none')+torch.tensor(1))
        losses = {}
        losses['loss_point'] = (loss_bbox.sum() / num_points)

        return losses

    def loss_nums(self, outputs, targets, indices, num_points):
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # accumulate MAE, MSE
        focal_l1=Focal_L1()
        loss=focal_l1(predict_cnt,num_points)
        losses = {}
        losses['loss_nums'] =loss
        # mae = abs(predict_cnt - gt_cnt)
        return losses
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'nums': self.loss_nums
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))
        # print(losses)
        return losses

# create the M2PLNet model

def build(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = load_backbone()
    model = M2PLNet(backbone, args.row, args.line)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_point': 0.00, 'loss_nums': 0.00001}
    losses = ['labels', 'points','nums']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion
