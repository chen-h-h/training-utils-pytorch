The `config.yaml` define the parameters used. The model, data_loader, loss_fn, optimizer and lr_scheduler can be defined in other file or use functions that come with PyTorch, which are given as strings. In this example, the model and dataloader are defined in `mnist_utils.py`, and the others are provided by PyTorch.

Run the script by the following command.
## Single-gpu training

```bash
CUDA_VISIBLE_DEVICES=1 python train_mnist.py --config config.yaml 
```

## Multi-gpu training
Assume use two GPUs.

```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 train_mnist.py --config config.yaml 
```