# Basics
import numpy as np
import pandas as pd
import pickle
import h5py
import os

from scripts.data_gen import DataGenerator
from keras.models import load_model
from glob import glob


name = "cnn_reg6"
classification = False
timesteps = 1
batch_size = 1
mode = "valid" # partition to eval on


# partitions
with open("../data/partition.pkl", 'rb') as f:
    partition = pickle.load(f)

# labels
target = pd.read_csv("../data/ERU_Scores_Ids_5-Scans_Validity-0_VisuallyScored.csv")
labels = target.set_index("StId").to_dict()["ERU.M2"]

label_converter = {0: 0, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6} if classification else {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}
labels = {key: label_converter[val] for key, val in labels.items()}

# model
path = sorted(glob("../output/"+name+"/weights/*.hdf5"))[-1]
model = load_model(path)
modelname = path.split("/")[-1].split(".hdf5")[0]
print("Loaded model:", modelname)

# data generator
dims = tuple([dim for dim in model.input_shape[3:]])
gen = DataGenerator(labels, partition, mode=mode, oversample=False,
                    classes=1, batch_size=batch_size, timesteps=timesteps, dims=dims)

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
df.columns = ["Actual", "Predicted", "Score"]

hit = np.sum(preds_int[:,0] == preds_int[:,1])
miss = np.sum(preds_int[:,0] != preds_int[:,1])
acc = hit / gen.__len__()

binary = df[["Actual", "Predicted"]].clip(0, 1, axis=0)
binary_hit = np.sum(binary["Actual"] == binary["Predicted"])
binary_miss = np.sum(binary["Actual"] != binary["Predicted"])
binary_acc = binary_hit / binary.shape[0]

binary0 = df[["Actual", "Predicted"]].replace([1,2,3,4], 1)
binary0 = binary0.loc[binary0['Actual'] == 0]
binary0_hit = np.sum(binary0["Actual"] == binary0["Predicted"])
binary0_miss = np.sum(binary0["Actual"] != binary0["Predicted"])
binary0_acc = binary0_hit / binary0.shape[0]

binary1 = df[["Actual", "Predicted"]].replace([0, 2, 3, 4], 0)
binary1 = binary1.loc[binary1['Actual'] == 1]
binary1_hit = np.sum(binary1["Actual"] == binary1["Predicted"])
binary1_miss = np.sum(binary1["Actual"] != binary1["Predicted"])
binary1_acc = binary1_hit / binary1.shape[0]

binary2 = df[["Actual", "Predicted"]].replace([0, 1, 3, 4], 0)
binary2 = binary2.loc[binary2['Actual'] == 2]
binary2_hit = np.sum(binary2["Actual"] == binary2["Predicted"])
binary2_miss = np.sum(binary2["Actual"] != binary2["Predicted"])
binary2_acc = binary2_hit / binary2.shape[0]

binary3 = df[["Actual", "Predicted"]].replace([0, 1, 2, 4], 0)
binary3 = binary3.loc[binary3['Actual'] == 3]
binary3_hit = np.sum(binary3["Actual"] == binary3["Predicted"])
binary3_miss = np.sum(binary3["Actual"] != binary3["Predicted"])
binary3_acc = binary3_hit / binary3.shape[0]

binary4 = df[["Actual", "Predicted"]].replace([0, 1, 2, 3], 0)
binary4 = binary4.loc[binary4['Actual'] == 4]
binary4_hit = np.sum(binary4["Actual"] == binary4["Predicted"])
binary4_miss = np.sum(binary4["Actual"] != binary4["Predicted"])
binary4_acc = binary4_hit / binary4.shape[0]

# save
directory = "../output/" + name + "/predictions/"
if not os.path.exists(directory):
    os.makedirs(directory)

savepath = directory + "/" + modelname + ".csv"
df.to_csv(savepath, index=False)

with open(directory + "/" + modelname + ".txt", "w") as txt:
    txt.write("{0}\n\n".format(modelname))
    txt.write("Overall\n")
    txt.write("hit  = {0}\n".format(round(hit, 2)))
    txt.write("miss = {0}\n".format(round(miss,2)))
    txt.write("acc  = {0}\n\n".format(round(acc,2)))
    txt.write("0 vs rest\n")
    txt.write("hit  = {0}\n".format(round(binary_hit,2)))
    txt.write("miss = {0}\n".format(round(binary_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary_acc,2)))
    txt.write("0 only\n")
    txt.write("hit  = {0}\n".format(round(binary0_hit,2)))
    txt.write("miss = {0}\n".format(round(binary0_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary0_acc,2)))
    txt.write("1 only\n")
    txt.write("hit  = {0}\n".format(round(binary1_hit,2)))
    txt.write("miss = {0}\n".format(round(binary1_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary1_acc,2)))
    txt.write("2 only\n")
    txt.write("hit  = {0}\n".format(round(binary2_hit,2)))
    txt.write("miss = {0}\n".format(round(binary2_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary2_acc,2)))
    txt.write("3 only\n")
    txt.write("hit  = {0}\n".format(round(binary3_hit,2)))
    txt.write("miss = {0}\n".format(round(binary3_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary3_acc,2)))
    txt.write("4 only\n")
    txt.write("hit  = {0}\n".format(round(binary4_hit,2)))
    txt.write("miss = {0}\n".format(round(binary4_miss,2)))
    txt.write("acc  = {0}\n\n".format(round(binary4_acc,2)))

# print summary
print(name)
print("Overall")
print("Correct preds:", hit)
print("Incorrect preds:", miss)
print("Accuracy:", acc)
print()
print("0 vs rest")
print("Correct preds:", binary_hit)
print("Incorrect preds:", binary_miss)
print("Accuracy:", binary_acc)