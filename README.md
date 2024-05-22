## Installation
```bash
git clone https://github.com/chen-h-h/training-utils-pytorch.git
```
And then add the path of training-utils-pytorch to the python environment variable.
```bash
echo 'export PYTHONPATH=/path/to/training-utils-pytorch:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```
Please replace the `/path/to/training-utils-pytorch` using yours. You can use it as a python package such as:

```python
from tup import Trainer, TrainingArgs, ConfigArgParser ...
```

## Getting Started
In the [examples](https://github.com/chen-h-h/training-utils-pytorch/tree/main/examples) folder, we provide two examples to show how to use our `tup` package.
- med_seg: A medical image segmentation task on chest X-ray (CXR) images.
- med_cls: A image classification task on MNIST dataset.

## Highlights
Take the example [med_seg](https://github.com/chen-h-h/training-utils-pytorch/tree/main/examples/med_seg) as an example.
- `tup` provides a Trainer, implementing tedious and repetitive training logic. And the components of training, such as model, dataloader, loss , etc., can all be defined in separate files. For excample, we defined the model, loss and dataloader in the files under [med_seg/medseg](https://github.com/chen-h-h/training-utils-pytorch/tree/main/examples/med_seg/medseg) folder.
- `tup` adopts hook mechanism to make Trainer more extensible. For example, we can use use class `HookBase` to save checkpoints and logs. Please refers to [hooks](https://github.com/chen-h-h/training-utils-pytorch/tree/main/tup/hooks).
- `tup` adopts YAML files to configure all components for training and hyper parameters. Please refers to [base_config.yaml]([/home/cl/projects/training-utils-pytorch/examples/med_seg/configs/base_config.yaml](https://github.com/chen-h-h/training-utils-pytorch/blob/main/examples/med_seg/configs/base_config.yaml)). Furthermore, though these parameters are defined in the configuration file, we can also modify them in the command line like this:
    ```bash
    python train_seg.py --config configs/base_config.yaml data.args.batch_size=128 optimizer.name=torch.optim.SGD
    ```
    By introducing new options, we initiate the training process with updated parameters, such as setting the batch_size to 128 instead of 64 and choosing SGD as the optimizer in place of Adam.

## Acknowledgments
The tup is developed on the basis of [core-pytorch-utils (cpu)](https://github.com/serend1p1ty/core-pytorch-utils). If you have no requirement to configure parameters using YAML files, you can turn to `cpu`, which is more mature and reliable. And we also refered [detectron2](https://github.com/facebookresearch/detectron2) and [transformers.Trainer](https://huggingface.co/docs/transformers/main/main_classes/trainer#api-reference%20][%20transformers.Trainer) of hugging face.