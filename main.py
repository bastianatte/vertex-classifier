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
epochs = [100, 120, 160, 180, 200, 220]
nFeats = 10
nVtxFeats = 9
nTrackFeats = 5
loss = binary_crossentropy
print("loss is : ", loss)
epochs_name = 'Epochs'
mod_name = '_Epochs'
path = sys.argv[2]
if not os.path.exists(path):
    os.makedirs(path)
for e in epochs:
    epoch_str = str(e)
    epochs_name = 'Epochs'
    mod_name = '_Epochs'
    ep = os.path.join(epoch_str+epochs_name)
    mer.execute(nSamples,
                nVtxFeats,
                nTrackFeats,
                e,
                loss,
                ep,
                test_size,
                path,
                ep,
                load_feats,
                models)



