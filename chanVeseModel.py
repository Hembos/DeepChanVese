from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import roi_align
from torchvision.models.detection.roi_heads import keypointrcnn_loss, keypointrcnn_inference
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn.functional as F

import torch

import numpy as np

from chanVeseHeads import ChanVeseModel

import matplotlib.pyplot as plt 

import time

def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Args:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob

def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    phi_targets = (mask_targets - 0.5) / 0.5

    l1_loss = F.l1_loss(mask_logits, phi_targets)

    mask_logits = mask_logits.sigmoid()

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy(
        mask_logits, mask_targets
    )
    return l1_loss + mask_loss

class RoiHeadsWithTSDF(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor)
        
        self.cv = ChanVeseModel(100)
        
    def __apply_chan_vese_train__(self, mask_logits, mask_proposals, gt_labels, gt_masks, mask_matched_idxs):
        features = mask_logits

        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

        labels = torch.cat(labels, dim=0)

        masks_phi = []
        for class_num, feature in zip(labels, features):
            x = feature[class_num]
            start = time.time()
            x = self.cv(x)
            end = time.time()
            print(end - start)
            masks_phi.append(x)

        masks_phi = torch.tensor(np.array(masks_phi),  dtype=torch.float32)

        # plt.imshow(masks_phi[0])
        # plt.show()
        masks_phi = masks_phi.to(torch.device('cuda:0'))

        loss = maskrcnn_loss(masks_phi, mask_proposals, gt_masks, gt_labels, mask_matched_idxs)

        return loss
    
    def __apply_chan_vese_eval__(self, mask_logits, labels):
        features = mask_logits
        masks_phi = []
        for class_num, feature in zip(labels, features):
            x = feature[class_num]
            x = self.cv(x)
            masks_phi.append(x)

        masks_phi = torch.tensor(np.array(masks_phi),  dtype=torch.float32)

        masks_phi = masks_phi.to(torch.device('cuda:0'))

        mask_prob = masks_phi.sigmoid()

        return mask_prob

    def forward(
        self,
        features, 
        proposals,  
        image_shapes,  
        targets=None,  
    ):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")
            
            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")


                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = self.__apply_chan_vese_train__(mask_logits, mask_proposals, gt_labels, gt_masks, pos_matched_idxs)
                # rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                # masks_probs = maskrcnn_inference(mask_logits, labels)
                masks_probs = self.__apply_chan_vese_eval__(mask_logits, labels[0])
                result[0]["masks"] = torch.reshape(masks_probs, (masks_probs.shape[0], 1, masks_probs.shape[1], masks_probs.shape[2]))
                # for mask_prob, r in zip(masks_probs, result):
                #     r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
      
def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False)

    mask_roi_pool = model.roi_heads.mask_roi_pool
    mask_head = model.roi_heads.mask_head
    mask_predictor = model.roi_heads.mask_predictor

    model.roi_heads = RoiHeadsWithTSDF(
        model.roi_heads.box_roi_pool,
        model.roi_heads.box_head,
        model.roi_heads.box_predictor,
        model.roi_heads.proposal_matcher.high_threshold,
        model.roi_heads.proposal_matcher.low_threshold,
        model.roi_heads.fg_bg_sampler.batch_size_per_image,
        model.roi_heads.fg_bg_sampler.positive_fraction,
        model.roi_heads.box_coder.weights,
        model.roi_heads.score_thresh,
        model.roi_heads.nms_thresh,
        model.roi_heads.detections_per_img,
        model.roi_heads.mask_roi_pool,
        model.roi_heads.mask_head,
        model.roi_heads.mask_predictor,
        model.roi_heads.keypoint_roi_pool,
        model.roi_heads.keypoint_head,
        model.roi_heads.keypoint_predictor
        )
    
    model.roi_heads.mask_roi_pool = mask_roi_pool
    model.roi_heads.mask_head = mask_head
    model.roi_heads.mask_predictor = mask_predictor


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model