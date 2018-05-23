# Basics
import numpy as np
import pandas as pd
import pickle
import h5py

from scripts.data_gen import DataGenerator
from keras.models import load_model
from glob import glob


name = "cnn_reg9"
classification = False
timesteps = 1
batch_size = 1
cropped = True
mode = "valid"


# partitions
with open("../data/partition.pkl", 'rb') as f:
    partition = pickle.load(f)

# labels
target = pd.read_csv("../data/ERU_Scores_Ids_5-Scans_Validity-0_VisuallyScored.csv")
labels = target.set_index("StId").to_dict()["ERU.M2"]

label_converter = {0: 0, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6} if classification else {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}
labels = {key: label_converter[val] for key, val in labels.items()}

# data generator
dims = (142, 322, 262) if cropped else (50, 146, 118)
gen = DataGenerator(labels, partition, mode=mode, oversample=False,
                    classes=1, batch_size=batch_size, timesteps=timesteps, dims=dims)

# model
path = sorted(glob("../output/"+name+"/weights/*.hdf5"))[-1]
print(path)

model = load_model(path)

# predict
preds_int = np.empty((gen.__len__(), 2), dtype=int)
preds_float = np.empty((gen.__len__(), 1), dtype=float)

for batch in range(gen.__len__()):
    X, y = gen.__getitem__(batch)  # select batch index
    y_ = model.predict_on_batch(X)

    preds_int[batch, 0] = y[0,0]
    preds_int[batch, 1] = np.round(y_[0, 0])
    preds_float[batch, 0] = y_[0, 0]

    if batch % 10 == 0 or batch % gen.__len__() == 0:
        print("Predicted batch:", batch, "/", gen.__len__())

# format output
df1 = pd.DataFrame(preds_int)
df2 = pd.DataFrame(preds_float).round(decimals=3)
df = pd.concat([df1, df2], axis=1)

hit = np.sum(preds_int[:,0] == preds_int[:,1])
miss = np.sum(preds_int[:,0] != preds_int[:,1])
acc = hit / gen.__len__()

# save
modelname = path.split("/")[-1].split(".hdf5")[0]
savepath = "../output/" + name + "/preds_" + modelname + ".csv"
df.to_csv(savepath, index=False, header=["Actual", "Predicted", "Score"])

with open("../output/" + name + "/eval.txt", "w") as txt:
    txt.write("{0}\n\n".format(modelname))
    txt.write("hit = {0}\n".format(hit))
    txt.write("miss = {0}\n".format(miss))
    txt.write("acc = {0}\n".format(acc))

print(name)
print("Correct preds:", hit)
print("Incorrect preds:", miss)
print("Accuracy:", acc)