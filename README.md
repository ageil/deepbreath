# DeepBreath

This project aims to predict emphysema extent in human lungs from CT scans by utilizing convolutional and recurrent neural networks.

## Setup

Clone the repository and run the following command from the command line:

`CUDA_VISIBLE_DEVICES=0 python main.py modelname timesteps batch_size max_epochs`

For instance, the command 

`CUDA_VISIBLE_DEVICES=0 python main.py CNN1 1 3 100`

will launch a training session on GPU #0 as a regression task with 1 time step and a batch size of 3 for 50 epochs.

A myriad of additional settings and hyperparameters may be found by running `python main.py --help`.