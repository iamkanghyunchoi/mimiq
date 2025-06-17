# MimiQ: Low-Bit Data-Free Quantization of Vision Transformer with Encouraging Inter-Head Attention Similarity

This folder contains the official implementation of MimiQ: Low-Bit Data-Free Quantization of Vision Transformer with Encouraging Inter-Head Attention Similarity.

## Requirements

* Python 3.9.18
* PyTorch 2.0.1
* Refer to the requirements.txt for other requirements

## Setup

We recommend using Python virtual environment to run this code.

You can install requirements with the command below.

```setup
pip install -r requirements.txt
mkdir -p /datasets
ln -s {YOUR IMAGENET FOLDER} /datasets/image
```

## Folder Structure

```
mimiq_code
├── main.py                            
├── option.py 
├── trainer.py
├── imagenet_{NETWORK}.hocon                    # Setting files
├── train.sh                                    # Train script
├── generate_data.sh                            # Synthetic data generation script
├── merge_dataset.sh                            # Merge generated data into the dataset
├── trainer.py
├── ...                                         # Utils
├── LICENSE.md
├── README.md
└── requirements.txt
```

## Data Generation

For synthetic dataset reconstruction, run the data generation script below:

```
./generate_dataset.sh MODEL_NAME NUM_IMGS SAVE_PREFIX SAVE_PATH
```

* MODLE_NAME : Target network architecture.
ViT architectures: vit\_{tiny|small|base}\_patch16\_224
DeiT architectures: deit\_{tiny|small|base}\_patch16\_224
Swin architectures: swin\_{tiny|small|base}\_patch4\_window7\_224
* NUM_IMGS : The number of synthetic image per GPU
* SAVE_PREFIX : This number will be added to the image number, use for multi-gpu generation
e.g., if SAVE_PREFIX=1000 and NUM_IMGS=100, generated images will have ID from 1000 to 1100.
* SAVE_PATH : Where to save generated images. 

and run merge_dataset.sh SAVE_PATH MODEL_NAME

## Training

For training, change the path of the validation set in .hocon file.
To quantize the model described in the paper, run the training script below

```quantize
train.sh CONF_PATH ID LR QW QA GAMMA DATA_PATH LR_POLICY LR_STEP AQ_MODE
```

* CONF_PATH: path to .hocon file
* ID : experiment ID, you can use proper unsigned integer, such as 1234, 5678, etc.
* LR : Learning rate, default=0.001
* QW, QA : Weight quantization bit and Activation quantization bit
* GAMMA : Attention head distillation coefficient, default = 10.0
* DATA_PATH : Synthetic dataset path
* LR_POLICY : Learning rate policy, default=multi_step
* LR_STEP : Learning rate decay policy, defualt=[50,100]
* AQ_MODE : Activation quantization method, you can use minmax or lsq, default=lsq


## License

This project is licensed under the terms of the GNU General Public License v3.0
