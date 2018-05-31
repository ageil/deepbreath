# Basics
import numpy as np
import pandas as pd
import argparse
import pickle
import os
from glob import glob
from sklearn.utils import class_weight

# Keras
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Custom
from scripts.data_gen import DataGenerator
from scripts.gapnet import tdist_gapnet
from scripts.TBCallbacks import TrainValTensorBoard


# Set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="name of the model")
parser.add_argument("timesteps", type=int, help="number of time steps")
parser.add_argument("batch_size", type=int, help="number of samples in each batch, if oversampling batch_size >= 3")
parser.add_argument("max_epochs", type=int, help="maximum number of epochs")
parser.add_argument("--classification", default=False, action='store_true', help="train as classification or regression problem")
parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate")
parser.add_argument("--train_samples", default=100, type=int, help="number of training samples used (debug=20, max=779)")
parser.add_argument("--valid_samples", default=333, type=int, help="number of validation samples used (max=333)")
parser.add_argument("--cropped", default=True, type=bool, help="cropped images or full size")
parser.add_argument("--downsample", default=1, type=int, help="input downsampling factor")
parser.add_argument("--droprate", default=0.0, type=float, help="proportion of units dropped in dropout layers")
parser.add_argument("--reg", default=0.0, type=float, help="regularization parameter used for L2 regularization")
parser.add_argument("--oversample", default=True, type=bool, help="oversample training data")
parser.add_argument("--flip", default=True, type=bool, help="randomly flip training data")
parser.add_argument("--shift", default=True, type=bool, help="randomly shift training data")
parser.add_argument("--base", type=str, help="name of model containing pretrained weights")
parser.add_argument("--base_version", default=1, type=int, help="version of saved weights to load, counting from last")
parser.add_argument("--trainable", default=False, action='store_true', help="make base weights trainable if using pretrained weights")
parser.add_argument("--version", action='version', version="DeepBreath v0.9")
args = parser.parse_args()

# name = 'test'
# timesteps = 5
# batch_size = 3
# learn_rate = 1e-3
# max_epochs = 30

if args.learn_rate > 0:
    optimizer = Adam(lr=args.learn_rate)
    opt = "Adam"
else:
    optimizer = Nadam()
    opt = "Nadam"

# Load data partitions
with open("./data/partition.pkl", 'rb') as f:
    partition = pickle.load(f)
partition["train"] = partition["train"][:args.train_samples]
partition["valid"] = partition["valid"][:args.valid_samples]

# Load labels
target = pd.read_csv("./data/ERU_Scores_Ids_5-Scans_Validity-0_VisuallyScored.csv")
labels = target.set_index("StId").to_dict()["ERU.M2"]

# Rescale labels
if args.classification:
    loss = "categorical_crossentropy"
    metrics = ['acc', 'mae', 'mse']
else:
    loss = "mae"
    metrics = ['acc', 'mae', 'msle']
label_converter = {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0} # combine 0 (no emph in scan) + 1 (no emph in region)
labels = {key: label_converter[val] for key, val in labels.items()}


# Calculate class weights
# weights only based on training data
train_labels = [label for key, label in labels.items() if key in partition["train"]]
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(train_labels),
                                                  y=train_labels)

# Create data generators
params = {
    'classes': 6 if args.classification else 1,
    'timesteps': args.timesteps,
    'batch_size': args.batch_size,
    'channels': 1,
    'cropped': args.cropped,
}
trainGen = DataGenerator(labels, partition, mode="train", oversample=args.oversample, flip=args.flip, shift=args.shift, **params)
validGen = DataGenerator(labels, partition, mode="valid", oversample=False, **params)

# Create model
paths = sorted(glob("./output/"+args.name+"/weights/*.hdf5"))
if len(paths) > 0:
    path = paths[-1]
    model = load_model(path)
    print("Loaded model:", path)
else:
    model = tdist_gapnet(classification=args.classification, timesteps=args.timesteps, cropped=args.cropped,
                         downsample=args.downsample, droprate=args.droprate, reg=args.reg)

    # Optionally load previous model with pretrained weights
    if args.base:
        print("Loading pretrained weights...")
        path = sorted(glob("./output/" + args.base + "/weights/*.hdf5"))[-args.base_version]

        pretrained = load_model(path)
        print("Loaded base model:", path)
        print("Trainable weights:", args.trainable)

        for i in range(45):
            if pretrained.layers[i].name == model.layers[i].name:
                weight = pretrained.layers[i].get_weights()
                model.layers[i].set_weights(weight)
                model.layers[i].trainable = args.trainable

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Created new model:", args.name)


# Setup output folder
directory = "./output/"+args.name+"/"
if not os.path.exists(directory):
    os.makedirs(directory)

    with open("./output/"+args.name+"/config.txt", "w") as txt:
        txt.write("name = {0}\n".format(args.name))
        txt.write("classification = {0}\n".format(args.classification))
        txt.write("timesteps = {0}\n".format(args.timesteps))
        txt.write("batch_size = {0}\n".format(args.batch_size))
        txt.write("learn_rate = {0}\n".format(args.learn_rate))
        txt.write("max_epochs= {0}\n".format(args.max_epochs))
        txt.write("downsample = {0}\n".format(args.downsample))
        txt.write("droprate = {0}\n".format(args.droprate))
        txt.write("reg = {0}\n".format(args.reg))
        txt.write("train_samples = {0}\n".format(args.train_samples))
        txt.write("valid_samples = {0}\n".format(args.valid_samples))
        txt.write("oversample = {0}\n".format(args.oversample))
        txt.write("cropped = {0}\n".format(args.cropped))
        txt.write("loss = {0}\n".format(loss))
        txt.write("opt = {0}\n".format(opt))
        if args.base:
            txt.write("base = {0}\n".format(args.base))
            txt.write("base_version = {0}\n".format(args.base_version))
            txt.write("trainable = {0}\n".format(args.trainable))

# Set callbacks
callbacks = []

# save model weights if best
savepath = directory + "weights/"
if not os.path.exists(savepath):
    os.makedirs(savepath)
modeldir = savepath + "epoch_{epoch:03d}-valloss_{val_loss:.2f}-valacc_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modeldir, monitor='val_loss', save_weights_only=False, save_best_only=True, mode='min', verbose=1)
callbacks.append(checkpoint)

# custom tensorboard logging
tensorboard= TrainValTensorBoard(log_dir=directory+"logs/", histogram_freq=0, write_graph=True, write_images=True) # custom TB writer object
callbacks.append(tensorboard) # add tensorboard logging

# Train model
hist = model.fit_generator(generator = trainGen,
                           validation_data = validGen,
                           class_weight = class_weights,
                           epochs = args.max_epochs,
                           callbacks=callbacks,
                           use_multiprocessing=True,
                           workers=5)

# Save final model
finalsave = savepath + "epoch_{0:03d}".format(args.max_epochs) + "-valloss_{val_loss:.2f}_{val_acc:.2f}" + "_final.hdf5"
model.save(finalsave, include_optimizer=True, overwrite=True)

# Dump history to disk
with open("./output/"+args.name+"/history.pkl", 'wb') as f:
    pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)
