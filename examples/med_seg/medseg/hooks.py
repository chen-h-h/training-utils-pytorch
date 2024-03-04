import logging
import random
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from typing import Dict

from .metrics import compute_dice
from tup import EvalHook

logger = logging.getLogger(__name__)


def test(model: nn.Module, test_loader: DataLoader, n_classes: int) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    dice = [0.0]*n_classes

    with torch.no_grad():
        for batch in test_loader:
            input, target = batch['input'].to(device), batch['target'].to(device)
            pred = model(input)

            pred = torch.argmax(pred, dim=1)
            dice_list = compute_dice(pred, target, n_classes)
            dice = [i+j*input.shape[0] for i,j in zip(dice, dice_list)]

    # round up to four decimal places
    dice_dict = {f'dice_{i}': round(value/len(test_loader.dataset),4) for i,value in enumerate(dice)}

    logger.info("\nTest set: Average dice: {} with {} cases".format(
        dice_dict, len(test_loader.dataset)))
    
    # visulize the prediction (Optional)
    show_sum = 4
    idx = random.randint(0, input.shape[0]-4)
    plt.figure(figsize=(6, 8))
    for i in range(show_sum):
        plt.subplot(show_sum, 3, 1 + i*3)
        plt.imshow(input[idx+i].cpu().squeeze(), cmap='gray')
        plt.title('image')
        plt.subplot(show_sum, 3, 2 + i*3)
        plt.imshow(target[idx+i].cpu().squeeze(), cmap='gray')
        plt.title('gt')
        plt.subplot(show_sum, 3, 3 + i*3)
        plt.imshow(pred[idx+i].cpu().squeeze(), cmap='gray')
        plt.title('pred')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig('output_dir/visualize.jpg', bbox_inches='tight')
    plt.close()

    return dice_dict

def test_hook(model, test_loader, n_classes):
    return EvalHook(1, lambda: test(model, test_loader, n_classes))