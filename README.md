# SwinLSTM: A new recurrent cell for spatiotemporal modeling (ICCV 2023 Oral)

This repository contains the official PyTorch implementation of the following paper:
**SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM**<br>
Paper: <br>

## Introduction
<div align="center"><img width="98%" src="./architecture.PNG" /></div>
Integrating CNNs and RNNs to capture spatiotemporal dependencies is a prevalent strategy for spatiotemporal prediction tasks. However, the property of CNNs to learn local spatial information decreases their efficiency in capturing spatiotemporal dependencies, thereby limiting their prediction accuracy. In this paper, we propose a new recurrent cell, SwinLSTM, which integrates Swin Transformer blocks and the simplified LSTM, an extension that replaces the convolutional structure in ConvLSTM with the self-attention mechanism. Furthermore, we construct a network with SwinLSTM cell as the core for spatiotemporal prediction. Without using unique tricks, SwinLSTM outperforms state-of-the-art methods on Human3.6m, TaxiBJ, KTH, and Moving MNIST datasets. In particular, it exhibits a significant improvement in prediction accuracy compared to ConvLSTM. We hope that SwinLSTM can serve as a solid baseline to promote the advancement of spatiotemporal prediction accuracy.


## Preparation

### Requirements
- python 3
- pytorch 1.6+ 
- opencv-python
- scikit-image
- lpips
- numpy




## Training the Model
`train.py` saves the weights in `--checkpoint_save_dir` and shows the training logs. 
To train the model, run following command:
Training example for Moving-MNIST
python train.py \
--dataset 'movingmnist' 
--train_data_dir 'enter_the_path' --valid_data_dir 'enter_the_path' \
--checkpoint_save_dir './checkpoints' 
--img_size 64 --img_channel 1 --memory_size 100 \
--short_len 10 --long_len 30 --out_len 30 
--batch_size 128 --lr 0.0002 --iterations 300000


Training example for KTH-Action
python train.py 
--dataset 'kth' \
--train_data_dir 'enter_the_path' --valid_data_dir 'enter_the_path' 
--checkpoint_save_dir './checkpoints' \
--img_size 128 --img_channel 1 --memory_size 100 
--short_len 10 --long_len 40 --out_len 40 
--batch_size 32 --lr 0.0002 --iterations 300000


Descriptions of training parameters are as follows:
- `--dataset`: training dataset (movingmnist or kth)  
- `--train_data_dir`: directory of training set  
- `--valid_data_dir`: directory of validation set
- `--checkpoint_save_dir`: directory for saving checkpoints
- `--img_size`: height and width of frame  
- `--img_channel`: channel of frame
- `--memory_size`: memory slot size
- `--short_len`: number of short frames  
- `--long_len`: number of long frames
- `--out_len`: number of output frames
- `--batch_size`: mini-batch size
- `--lr`: learning rate
- `--iterations`: number of total iterations
- Refer to `train.py` for the other training parameters

## Testing the Model 
`test.py` saves the predicted frames in `--test_result_dir` or evalute the performances.

To test the model, run following command:
Testing example for Moving-MNIST
python test.py 

--dataset 'movingmnist' --make_frame True \
--test_data_dir 'enter_the_path' --test_result_dir 'enter_the_path' 
--checkpoint_load_file 'enter_the_path' \
--img_size 64 --img_channel 1 --memory_size 100 
--short_len 10 --out_len 30 
--batch_size 8


Testing example for KTH-Action
python test.py 

--dataset 'kth' --make_frame True 
--test_data_dir 'enter_the_path' --test_result_dir 'enter_the_path' 
--checkpoint_load_file 'enter_the_path' 
--img_size 128 --img_channel 1 --memory_size 100 \
--short_len 10 --out_len 40 
--batch_size 8

Descriptions of testing parameters are as follows:
- `--dataset`: test dataset (movingmnist or kth)
- `--make_frame`: whether to generate predicted frames  
- `--test_data_dir`: directory of test set
- `--test_result_dir`: directory for saving predicted frames 
- `--checkpoint_load_file`: file path for loading checkpoint
- `--img_size`: height and width of frame
- `--img_channel`: channel of frame  
- `--memory_size`: memory slot size
- `--short_len`: number of short frames
- `--out_len`: number of output frames 
- `--batch_size`: mini-batch size
- Refer to `test.py` for the other testing parameters

## Citation
If you find this work useful in your research, please cite the paper:
@inproceedings{lee2021video,
title={Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning},
author={Lee, Sangmin and Kim, Hak Gu and Choi, Dae Hwi and Kim, Hyung-Il and Ro, Yong Man},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2021}
}
