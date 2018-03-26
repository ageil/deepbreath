# Basics
import numpy as np
import pandas as pd
import os
import sys
import pickle
import h5py
from sklearn.utils import class_weight

# Keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

# Custom
from data_gen import DataGenerator
from unet import tdist_unet
from TBCallbacks import TrainValTensorBoard


# Set hyperparameters
# timesteps = int(sys.argv[1])
# batch_size = int(sys.argv[2])
timesteps = 1
batch_size = 4
learn_rate = 1e-4
max_epochs = 50
name = "single"
downsample = 2


# Load data partitions
with open("./data/partition.pkl", 'rb') as f:
    partition = pickle.load(f)

# Load labels
target = pd.read_csv("./data/ERU_Scores_Ids_5-Scans_Validity-0_VisuallyScored.csv")
labels = target.set_index("StId").to_dict()["ERU.M2"]

# Rescale labels; combine 0+1 as 0 = no emph in scan, 1 = no emph in region
label_converter = {0: 0.0, 1: 0.0, 2: 0.03, 3: 0.155, 4: 0.38, 5: 0.63, 6: 0.88}
labels = {key: label_converter[val] for key, val in labels.items()}


# Calculate class weights
# weights only based on training data
train_labels = [label for key, label in labels.items() if key in partition["train"]]
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(train_labels),
                                                  y=train_labels)

# Create data generators
trainGen = DataGenerator(name+"_train", batch_size=batch_size, timesteps=timesteps,
                         channels=1, dim_x=142, dim_y=322, dim_z=262, shuffle=True)
validGen = DataGenerator(name+"_valid", batch_size=batch_size, timesteps=timesteps,
                         channels=1, dim_x=142, dim_y=322, dim_z=262, shuffle=False)

trainGen = trainGen.generate(labels, partition["train"])
validGen = validGen.generate(labels, partition["valid"])


# Create model
model = tdist_unet(timesteps=timesteps, downsample=downsample)
model.compile(optimizer=Adam(lr = learn_rate),
              loss='mean_absolute_error',
              metrics=['accuracy', 'mae', 'mse'])


# Set callbacks
callbacks_list = []

directory = "./models/"+name+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
modeldir = directory + "epoch_{epoch:02d}-valacc_{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(modeldir, monitor='val_acc', save_weights_only=False, save_best_only=True, mode='max', verbose=1)
callbacks_list.append(checkpoint) # saves model weights
# Load model weights using: model.load_weights(modeldir)

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
tensorboard= TrainValTensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True) # custom TB writer object
callbacks_list.append(tensorboard) # add tensorboard logging
# Access tensorboard using: tensorboard --logdir path_to_current_dir/graph


# Train model
hist = model.fit_generator(generator = trainGen,
                           steps_per_epoch = len(partition["train"])//batch_size,
                           validation_data = validGen,
                           validation_steps = len(partition["valid"])//batch_size,
                           class_weight = class_weights,
                           epochs = max_epochs,
                           callbacks=callbacks_list)


# Dump history to disk
if not os.path.exists("../output/"):
    os.makedirs("../output/")
with open("../output/"+name+"_history.pkl", 'wb') as f:
    pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)