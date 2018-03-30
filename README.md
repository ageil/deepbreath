# DeepBreath

This project aims to predict emphysema presence in human lungs from CT scans by utilizing convolutional and recurrent neural networks.

## Setup

Clone the repository and run the following command from the command line:

`CUDA_VISIBLE_DEVICES=0 python main.py classification num_timesteps batch_size learn_rate max_epochs downsample`

For instance, the command 

`CUDA_VISIBLE_DEVICES=0 python main.py True 1 5 1e-4 50 2`

will launch a training session on GPU #0 as a classification task with 1 time step, a batch size of 5, learning rate of 1e-4 for 50 epochs with the initial images downscaled by a factor of 2. 

Note that setting the learning rate to 0 will change the optimizer from Adam to the Nesterov Adam optimizer.

## Data summary

| Label | Full  | Training | Validation | Test   |
|:-----:| -----:| --------:| ----------:| ------:|
| 0+1   | 70.7% | 72.27%   | 68.17%     | 69.31% |
| 2     | 16.2% | 15.4%    | 18.62%     | 15.52% |
| 3     | 9.22% | 8.34%    | 9.91%      | 10.83% |
| 4     | 2.88% | 2.95%    | 2.4%       | 3.25%  |
| 5     | 0.79% | 0.77%    | 0.9%       | 0.72%  |
| 6     | 0.22% | 0.26%    | 0.0%       | 0.36%  |
