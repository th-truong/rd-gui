from training.vrb_model import vrb_training_model, rcnn

from collections import OrderedDict
import warnings
import itertools

from torch.jit.annotations import Tuple, List, Dict, Optional
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from pathlib import Path

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.faster_rcnn import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def create_rd_model(mrcnn_model_path, min_size=800, max_size=1333,
                    mask_mold_filters=8,
                    num_classes=201, num_rel_classes=100,
                    obj_score_thresh=0.25,
                    FPN_filters=256,
                    soft_attention_filters=256,
                    bottleneck_filters=16,
                    hidden_layer_sizes=[128, 64, 128],
                    pool_sizes={'0': 50,
                                '1': 25,
                                '2': 12,
                                '3': 12,
                                'pool': 6},
                    keys=['0', '1', '2', '3', 'pool']
                    ):

    pool_sizes = {key: _int_to_tuple(value) for value, key in pool_sizes.items()}

    mrcnn_model = create_mrcnn_model(mrcnn_model_path, num_classes)

    mask_transform = RdMaskTransform(min_size=min_size, max_size=max_size)
    mask_mold = RdMoldMaskInputs(mask_mold_filters=mask_mold_filters)
    soft_attention = RdSoftAttentionMechanism(FPN_filters=FPN_filters,
                                              mask_mold_filters=mask_mold_filters,
                                              soft_attention_filters=soft_attention_filters,
                                              bottleneck_filters=bottleneck_filters,
                                              keys=keys)
    classifier_head = MLPClassifierHead(pool_sizes=pool_sizes,
                                        num_classes=num_rel_classes,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        bottleneck_filters=bottleneck_filters,
                                        keys=keys)

    model = RdModel(mrcnn_model, mask_transform, mask_mold, soft_attention, classifier_head)

    return model


def create_rd_training_models(mrcnn_model_path, min_size=800, max_size=1333,
                              mask_mold_filters=8,
                              num_classes=201, num_rel_classes=100,
                              obj_score_thresh=0.25,
                              FPN_filters=256,
                              soft_attention_filters=256,
                              bottleneck_filters=16,
                              hidden_layer_sizes=[128, 64, 128],
                              pool_sizes={'0': 50,
                                          '1': 25,
                                          '2': 12,
                                          '3': 12,
                                          'pool': 6},
                              keys=['0', '1', '2', '3', 'pool']
                              ):

    pool_sizes = {str(key): _int_to_tuple(value) for key, value in pool_sizes.items()}
    mrcnn_model = create_mrcnn_model(mrcnn_model_path, num_classes)

    mask_transform = RdMaskTransform(min_size=min_size, max_size=max_size)
    mask_mold = RdMoldMaskInputs(mask_mold_filters=mask_mold_filters)
    soft_attention = RdSoftAttentionMechanism(FPN_filters=FPN_filters,
                                              mask_mold_filters=mask_mold_filters,
                                              soft_attention_filters=soft_attention_filters,
                                              bottleneck_filters=bottleneck_filters,
                                              keys=keys)
    classifier_head = MLPClassifierHead(pool_sizes=pool_sizes,
                                        num_classes=num_rel_classes,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        bottleneck_filters=bottleneck_filters,
                                        keys=keys)

    model = RdTrainModel(mask_transform, mask_mold, soft_attention, classifier_head)

    return model, mrcnn_model


def _int_to_tuple(value):
    value = int(value)
    return (value, value)


def create_mrcnn_model(mrcnn_model_path, num_classes=201):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)
    model = RDFasterRCNN(backbone, num_classes)
    model_dict = torch.load(mrcnn_model_path)
    model.load_state_dict(model_dict['model'])
    return model


class RdModel(nn.Module):
    # must manually handle batches because we are considering all relationships in one image to be a "batch"
    # for now only batch_size=1 has been tested
    def __init__(self, mrcnn, mask_transform, mask_mold, soft_attention, classifier_head, obj_score_thresh=0.25,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(RdModel, self).__init__()
        self.mrcnn = mrcnn
        self.mask_transform = mask_transform
        self.soft_attention = soft_attention
        self.mask_mold = mask_mold
        self.classifier_head = classifier_head
        self.device = device
        self.obj_score_thresh = obj_score_thresh
        self.device_float_one = torch.Tensor([1.0]).to(self.device)

    def forward(self, img, targets=None, sub_inputs=None, obj_inputs=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        losses, detections, features = self.mrcnn(img, targets)

        if self.training:
            sub_images, obj_images = self.mask_transform(sub_inputs, obj_inputs)
        else:
            sub_images, obj_images = self.mask_transform(sub_inputs, obj_inputs)
            # TODO: implement feeding mrcnn to soft attention after training is finished

        soft_mask_inputs = []
        attention_features = []
        for i in range(len(sub_images)):
            sub_box_masks = torch.stack(sub_images[i])
            obj_box_masks = torch.stack(obj_images[i])
            soft_mask_inputs.append(self.mask_mold(sub_box_masks, obj_box_masks))
            attention_features.append(self.soft_attention(features, soft_mask_inputs[i]))
        out = []
        for attention_feature in attention_features:
            out.append(self.classifier_head(attention_feature))

        return out

    def _create_box_image(self, box_coords, img_shape) -> torch.tensor:
        mask = torch.zeros(img_shape[0:2], dtype=torch.float32)
        box_coords = [torch.round(x) for x in box_coords]
        mask[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]] = self.device_float_one
        return mask


class RdTrainModel(nn.Module):
    # must manually handle batches because we are considering all relationships in one image to be a "batch"
    # for now only batch_size=1 has been tested
    def __init__(self, mask_transform, mask_mold, soft_attention, classifier_head, obj_score_thresh=0.25,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(RdTrainModel, self).__init__()
        self.mask_transform = mask_transform
        self.soft_attention = soft_attention
        self.mask_mold = mask_mold
        self.classifier_head = classifier_head
        self.device = device
        self.obj_score_thresh = obj_score_thresh
        self.device_float_one = torch.Tensor([1.0]).to(self.device)

    def forward(self, features, targets=None, sub_inputs=None, obj_inputs=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            sub_images, obj_images = self.mask_transform(sub_inputs, obj_inputs)
        else:
            sub_images, obj_images = self.mask_transform(sub_inputs, obj_inputs)
            # TODO: implement feeding mrcnn to soft attention after training is finished

        sub_box_masks = torch.stack(sub_images)
        obj_box_masks = torch.stack(obj_images)
        soft_mask_inputs = (self.mask_mold(sub_box_masks, obj_box_masks))
        attention_features = (self.soft_attention(features, soft_mask_inputs))
        out = self.classifier_head(attention_features)

        return out

    def _create_box_image(self, box_coords, img_shape) -> torch.tensor:
        mask = torch.zeros(img_shape[0:2], dtype=torch.float32)
        box_coords = [torch.round(x) for x in box_coords]
        mask[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]] = self.device_float_one
        return mask


class MLPClassifierHead(nn.Module):

    def __init__(self, pool_sizes, num_classes, hidden_layer_sizes=[256, 128, 256],
                 bottleneck_filters=32, keys=['0', '1', '2', '3', 'pool']):
        super(MLPClassifierHead, self).__init__()

        self.keys = keys

        num_features = 0
        for pool_size in pool_sizes.values():
            area = pool_size[0] * pool_size[1]
            num_features += area * bottleneck_filters

        self.num_features = num_features

        self.avg_pool = nn.ModuleDict({key: nn.AdaptiveAvgPool2d(pool_sizes[key]) for key in self.keys})

        self.fc0 = nn.Linear(self.num_features, hidden_layer_sizes[0])
        self.fc0dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc1dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])

        self.fc_classifier = nn.Linear(hidden_layer_sizes[2], num_classes)

    def forward(self, attention_features):
        pool_outs = [self.avg_pool[key](attention_features[key])
                     for key in self.keys]

        pool_outs = [pool_out.flatten(start_dim=1) for pool_out in pool_outs]

        concat = torch.cat(tuple(pool_outs), 1)

        fc0_out = F.relu(self.fc0(concat))
        fc0drop_out = self.fc0dropout(fc0_out)

        fc1_out = F.relu(self.fc1(fc0drop_out))
        fc1drop_out = self.fc1dropout(fc1_out)

        fc2_out = F.relu(self.fc2(fc1drop_out))

        if self.training:
            x = self.fc_classifier(fc2_out)
        else:
            x = torch.sigmoid(self.fc_classifier(fc2_out))
        # TODO: remove this for training, BCEwithlogitsloss applies a sigmoid before calculating loss

        return x


class RdMaskTransform(nn.Module):
    def __init__(self, min_size, max_size):
        super(RdMaskTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def forward(self,
                sub_inputs,
                obj_inputs
                ):
        person_image_list = self.process_inputs(sub_inputs)
        obj_image_list = self.process_inputs(obj_inputs)
        return person_image_list, obj_image_list

    def process_inputs(self, image_batch):
        image_batch = [img for img in image_batch]
        for j in range(len(image_batch)):
            image = image_batch[j]

            image = self.resize(image)
            image_batch[j] = image
        return image_batch

    def resize(self, image):
        h, w = image.shape[-2:]
        size = float(self.min_size[-1])
        image = self._resize_image_and_masks(image, size, float(self.max_size))

        return image

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _resize_image_and_masks(self, image, self_min_size, self_max_size):
        # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        scale_factor = self_min_size / min_size
        if max_size * scale_factor > self_max_size:
            scale_factor = self_max_size / max_size
        image = F.interpolate(image[:, None].float(), scale_factor=scale_factor, recompute_scale_factor=True)[:, 0].byte()

        return image


class RdMoldMaskInputs(nn.Module):

    def __init__(self, mask_mold_filters=8, mask_mold_ablation=False):
        super(RdMoldMaskInputs, self).__init__()
        # TODO: probably significantly reduce the number of inplanes here to reduce params significantly
        self.mask_mold_ablation = mask_mold_ablation
        self.mask_mold_filters = mask_mold_filters
        self.conv1 = nn.Conv2d(3, self.mask_mold_filters, kernel_size=7, stride=2, padding=3,
                               bias=False)

        # if not self.mask_mold_ablation:
        #     self.bn1 = nn.BatchNorm2d(self.mask_mold_filters)  # probably remove this batch norm
        # else:
        #     self.bn1 = nn.BatchNorm2d(3)  # probably remove this batch norm
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, person_image_list, obj_image_list):

        combined_mask = torch.add(person_image_list, obj_image_list)
        soft_mask = torch.cat((person_image_list, obj_image_list, combined_mask), 1)
        if not self.mask_mold_ablation:
            x = self.conv1(soft_mask.float())
        else:
            x = soft_mask.float()
        # x = self.bn1(x)
        x = self.relu(x)
        soft_mask_input = self.maxpool(x)

        return soft_mask_input


class RdSoftAttentionMechanism(nn.Module):
    def __init__(self, FPN_filters=256, mask_mold_filters=8, soft_attention_filters=256, bottleneck_filters=16,
                 keys=['0', '1', '2', '3', 'pool']):
        # keys must be a subset (or the entire set) of the keys in features from the FPN
        super(RdSoftAttentionMechanism, self).__init__()
        self.FPN_filters = FPN_filters
        self.mask_mold_filters = mask_mold_filters
        self.soft_attention_filters = soft_attention_filters
        self.bottleneck_filters = bottleneck_filters
        self.keys = keys

        self.conv_layers = nn.ModuleDict({key: nn.Conv2d(self.mask_mold_filters, self.soft_attention_filters, kernel_size=3, padding=1) for key in keys})
        self.relu = nn.ModuleDict({key: nn.ReLU(inplace=True) for key in keys})

        self.bottleneck_conv = nn.ModuleDict({key: nn.Conv2d(self.FPN_filters + self.soft_attention_filters,
                                                             self.bottleneck_filters, kernel_size=1, bias=False) for key in keys})
        self.bottleneck_relu = nn.ModuleDict({key: nn.ReLU(inplace=True) for key in keys})

    def forward(self, features, soft_mask_input):
        feature_shapes = {key: features[key].shape[-2:] for key in self.keys}
        num_rels = len(soft_mask_input)
        # for resnet50fpn feature_shapes contains (batch, filters, height, width) for corresponding feature key
        # the default for resnet50fpn is filters=256 and 5 keys in the features.

        masks_reshaped = {key: self._resize_soft_mask(soft_mask_input, feature_shapes[key])
                          for key in self.keys}

        masks_reshaped_filtered = {key: self.conv_layers[key](masks_reshaped[key])
                                    for key in self.keys}

        relu_out = {key: self.relu[key](masks_reshaped_filtered[key])
                    for key in self.keys}

        attention_features = {key: torch.cat((relu_out[key], torch.cat((features[key],)*num_rels)), 1)
                              for key in self.keys}

        attention_features_bottleneck = {key: self.bottleneck_conv[key](attention_features[key])
                                         for key in self.keys}
        attention_features_bottleneck_relu = {key: self.bottleneck_relu[key](attention_features_bottleneck[key])
                                              for key in self.keys}

        return attention_features_bottleneck_relu

    def _resize_soft_mask(self, mask, shapes):
        reshaped_mask = F.interpolate(mask.float(), size=(shapes[0], shapes[1]))
        return reshaped_mask


class RDFasterRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None):
        # modifies the FasterRCNN forward function to also return the backbone features
        super(RDFasterRCNN, self).__init__(backbone, num_classes)

    def eager_outputs(self, losses, detections, features):
        if self.training:
            return losses, features

        return detections, features

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections, features
