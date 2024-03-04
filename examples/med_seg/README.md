In this example, we implement a medical image segmentation tasks. Specifically, we segment the lung region from the chest X-ray (CXR) images.


## 1. Prepare data
Download the [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu/data) and put it in the `examples/med_seg/data` folder.

```bash
cd examples/med_seg/data
# We need the single quote because there are spaces in the name of file or folder.
unzip 'COVID-QU-Ex Dataset.zip' -d .
mv 'COVID-QU-Ex_Dataset/Infection Segmentation Data/Infection Segmentation Data' .
```
The dataset is large, so we use only the **Normal** data of `Infection Segmentation Data` part of it.

## 2. Training
When we have the data ready, the folder structure should look like:
```bash
examples/med_seg/
├── configs                          # The config files
│   └── base_config.yaml
├── data
│   └── Infection Segmentation Data  # The dataset
│       ├── Test
│       ├── Train
│       └── Val
├── medseg                           # The custom modules and functions
│   ├── data_loader.py
│   ├── hooks.py
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   └── model.py
├── README.md
└── train_seg.py                     # The main training function
```

Run the script by the following command.
### 2.1 Single-gpu training

```bash
cd examples/med_seg/
CUDA_VISIBLE_DEVICES=0 python train_seg.py --config configs/base_config.yaml
```

### 2.2 Multi-gpu training
Assume use two GPUs.

```bash
cd examples/med_seg/
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_seg.py --config configs/base_config.yaml
```
## Visualization
Observing the training loss:

```bash 
tensorboard --logdir examples/med_seg/output_dir/tb_logs
```
Observing the prediction in the test round in `output_dir/visualize.jpg`