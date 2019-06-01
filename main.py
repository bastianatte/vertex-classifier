from keras.losses import logcosh, binary_crossentropy, mean_squared_error,\
    mean_absolute_error
import os
import sys
import InDetMergeVtx as mer 

models = {
    "vertex": 1,
    "tracks": 0,
    "vertex_lstm": 0,
    "tracks_lstm": 0,
    "conc": 0,
    "add": 0,
    "conc_best": 0
}

load_feats = {"vertex": 1, "tracks": 0}

nSamples = 110000
test_size = 20000
epochs = 5
nFeats = 10
nVtxFeats = 9
nTrackFeats = 5
loss = binary_crossentropy
print("loss is : ", loss)
epochs_name = '5Epochs'
mod_name = '_5Epochs'
path = sys.argv[2]
if not os.path.exists(path):
    os.makedirs(path)
mer.execute(nSamples,
            nVtxFeats,
            nTrackFeats,
            epochs,
            loss,
            epochs_name,
            test_size,
            path,
            mod_name,
            load_feats,
            models)



