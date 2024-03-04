import torch
import numpy as np
from typing import Union, List, Dict

def dice_oneclass(
        pred: Union[torch.Tensor, np.ndarray], 
        gt: Union[torch.Tensor, np.ndarray]
    ) -> float:
    """ 
        dice = (2 * (pred ∩ gt)) / (pred + gt)
        pred: BHW
        gt: BHW
    """
    assert len(gt.shape)==3

    eps=1e-5
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice =  (2 * intersection + eps) / (unionset + eps)

    return float(dice.sum() / N)

def dice_multiclass(
        pred: Union[torch.Tensor, np.ndarray], 
        gt: Union[torch.Tensor, np.ndarray],
        num_class: int,
    ) -> List[float]:


    dice_list = []
    for i in range(num_class):
        if isinstance(pred, torch.Tensor): 
            pred_class = (pred==i).type(torch.uint8)
            gt_class = (gt==i).type(torch.uint8)
        else:
            pred_class = (pred==i).astype(np.uint8)
            gt_class = (gt==i).astype(np.uint8)

        dice_list.append(dice_oneclass(pred_class, gt_class))

    return dice_list

def compute_dice(
        pred: Union[torch.Tensor, np.ndarray], 
        gt: Union[torch.Tensor, np.ndarray],
        num_class: int,
    ) -> List[float]:

    dice_list = []
    for i in range(num_class):
        if isinstance(pred, torch.Tensor): 
            pred_class = (pred==i).type(torch.uint8)
            gt_class = (gt==i).type(torch.uint8)
        else:
            pred_class = (pred==i).astype(np.uint8)
            gt_class = (gt==i).astype(np.uint8)

        dice_list.append(dice_oneclass(pred_class, gt_class))

    return dice_list