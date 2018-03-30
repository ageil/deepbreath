# Basics
import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.utils import class_weight

# Keras
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint

# Custom
from scripts.data_gen import DataGenerator
from scripts.unet import tdist_unet
from scripts.TBCallbacks import TrainValTensorBoard


# Set hyperparameters
name = sys.argv[1]
classification = bool(sys.argv[2])
timesteps = int(sys.argv[3])
batch_size = int(sys.argv[4])
learn_rate = float(sys.argv[5])
max_epochs = int(sys.argv[6])
downsample = int(sys.argv[7])
droprate = float(sys.argv[8])
overfit = bool(sys.argv[9])

# name = "test"
# classification = False
# timesteps = 1
# batch_size = 5
# learn_rate = 1e-4
# max_epochs = 50
# downsample = 2
# droprate = 0.5
# overfit = True

if learn_rate > 0:
    optimizer = Adam(lr = learn_rate)
    opt = "Adam"
else:
    optimizer = Nadam()
    opt = "Nadam"


# Load data partitions
with open("./data/partition.pkl", 'rb') as f:
    partition = pickle.load(f)

if overfit:
    partition["train"] = partition["train"][:20]
    partition["valid"] = partition["valid"][:20]

# Load labels
target = pd.read_csv("./data/ERU_Scores_Ids_5-Scans_Validity-0_VisuallyScored.csv")
labels = target.set_index("StId").to_dict()["ERU.M2"]

# Rescale labels
if classification:
    # combine 0+1 as 0 = no emph in scan, 1 = no emph in region
    label_converter = {0: 0, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    loss = "categorical_crossentropy"
    metrics = ['acc', 'mae', 'mse']
else:
    # Rescale to [0;1]; merge category 0+1
    label_converter = {0: 0.0, 1: 0.0, 2: 0.03, 3: 0.155, 4: 0.38, 5: 0.63, 6: 0.88}
    loss = "mean_absolute_error"
    metrics = ['acc', 'mae', 'mse']
labels = {key: label_converter[val] for key, val in labels.items()}


# Calculate class weights
# weights only based on training data
train_labels = [label for key, label in labels.items() if key in partition["train"]]
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(train_labels),
                                                  y=train_labels)

# Create data generators
trainGen = DataGenerator("train", classification=classification, batch_size=batch_size,
                         timesteps=timesteps, channels=1, dim_x=142, dim_y=322, dim_z=262, shuffle=True)
validGen = DataGenerator("valid", classification=classification, batch_size=batch_size,
                         timesteps=timesteps, channels=1, dim_x=142, dim_y=322, dim_z=262, shuffle=True)

trainGen = trainGen.generate(labels, partition["train"])
validGen = validGen.generate(labels, partition["valid"])


# Create model
model = tdist_unet(classification=classification, timesteps=timesteps, downsample=downsample, droprate = droprate)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

# Setup output folder
directory = "./output/"+name+"/"
if not os.path.exists(directory):
    os.makedirs(directory)

with open("./output/"+name+"/config.txt", "w") as txt:
    txt.write("name = {0}\n".format(name))
    txt.write("timesteps = {0}\n".format(timesteps))
    txt.write("batch_size = {0}\n".format(batch_size))
    txt.write("learn_rate = {0}\n".format(learn_rate))
    txt.write("max_epochs= {0}\n".format(max_epochs))
    txt.write("downsample = {0}\n".format(downsample))
    txt.write("droprate = {0}\n".format(droprate))
    txt.write("loss = {0}\n".format(loss))
    txt.write("opt = {0}\n".format(opt))

# Set callbacks
callbacks_list = []

# save model weights if best
modeldir = directory + "epoch_{epoch:02d}-valacc_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modeldir, monitor='val_acc', save_weights_only=False, save_best_only=True, mode='max', verbose=1)
callbacks_list.append(checkpoint)

# early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=2) # stop if no improvements after 2 epochs
# callbacks_list.append(early_stopping)

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
tensorboard= TrainValTensorBoard(log_dir="./output/"+name+"/logs", histogram_freq=0, write_graph=True, write_images=True) # custom TB writer object
callbacks_list.append(tensorboard) # add tensorboard logging


# Train model
hist = model.fit_generator(generator = trainGen,
                           steps_per_epoch = len(partition["train"])//batch_size,
                           validation_data = validGen,
                           validation_steps = len(partition["valid"])//batch_size,
                           class_weight = class_weights,
                           epochs = max_epochs,
                           callbacks=callbacks_list)


# Dump history to disk
with open("./output/"+name+"/logs/history.pkl", 'wb') as f:
    pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)
