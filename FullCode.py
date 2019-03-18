 

### DATASET VARIABLES ###

DATASET_VTX = [ "sumpt", "trks", "W2", "w4", "chi_squared", "ch_dof", "isMerge"] #best variables_new  ###"sumpt","sumpt2",      
LAB_VTX = "isMerge"
DATASET_TRK = ["pt","eta","phi","d0","z0","isMerge_trk_2"]
LAB_TRK = "isMerge_trk_2"
ROOTFILE = '/home/atlas/sspinali/Qual_task/Data/test_RandomForest_full.root'
#ROOTFILE_VTX = '/home/spinali/lapzhd/Qual_task/Data/test_vertices_few_events.root'
#ROOTFILE_TRK = '/home/spinali/lapzhd/Qual_task/Data/test_tracks_few_events.root'


## load data method ##
def load_data(path_to_load, data, lab, nSample, nFeat):
    import uproot 
    import pandas
    import numpy as np
    
    f = uproot.open(path_to_load)
    tree = f['Merged_Analysis;1']
    dataset = tree.pandas.df(data)
    features = dataset.loc[:, dataset.columns != lab].values
    labels = dataset.loc[:, lab].values

    features = np.resize(features, (nSample, nFeat))
    labels = np.resize(labels, (nSample,))

    return features, labels


## vertices model ##
def vertex_model():   
    ### keras ###                                                                                                                     
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.activations import sigmoid, tanh
    from keras.optimizers import Adam, Nadam, sgd
    from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
    
    model = Sequential()
    
    model.add(Dense(units=9,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dropout(0.4))
    
    model.add(Dense(units=512,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=256,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=128,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))
    
    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


### tracks model ###
def tracks_model():
    ### keras ###
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.activations import sigmoid, tanh
    from keras.optimizers import Adam, Nadam, sgd
    from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
    from keras.callbacks import TensorBoard
    
    model = Sequential()
    
    model.add(Dense(units=5,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    #model.add(Dropout(0.2))
    
    model.add(Dense(units=512,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=256,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=128,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=64,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))
    
    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
        
    return model

def LSTM_model():
    ### keras ###
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from keras.activations import sigmoid
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    
    model = Sequential()

    model.add(LSTM( 1024, input_shape=(1,5), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM( 1024, input_shape=(1,5), return_sequences=True))
    model.add(LSTM( 256, input_shape=(1,5), return_sequences=True))
    #model.add(LSTM( 64, input_shape=(1,5), return_sequences=True))
    #model.add(LSTM( 16, input_shape=(1,5), return_sequences=True))
    #model.add(LSTM( 4, input_shape=(1,5), return_sequences=True))

    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
 
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])

    return model



### training steps method ###
def train_model(model, features, labels, epochs):
    ### keras ### 
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.activations import sigmoid, tanh
    from keras.optimizers import Adam, Nadam, sgd
    from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
    from keras.callbacks import TensorBoard
    #### sklearn ####
    from sklearn.model_selection import train_test_split
    

    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        labels, 
                                                        test_size=0.2, 
                                                        random_state=42)


    history = model.fit(X_train, y_train,
                        epochs=epochs, 
                        #batch_size=batch_size,
                        validation_data=(X_test, y_test)) 
                        #show_accuracy=True,
                        #verbose=1)

    print('Testing...')
    scores = model.evaluate(X_test, y_test) #batch_size=batch_size, verbose=1, show_accuracy=True)    
    print("SCORE = %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("SCORE = %s: %.2f%%" % (model.metrics_names[0], scores[0]*100))

    
    return history, scores, X_test, y_test, model


def conc_method(features_vtx, labels_vtx, features_trk, labels_trk, epochs):
    ### vector ###
    import uproot
    import pandas
    import numpy as np
    ### keras ###
    import keras
    from keras import layers, models
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Input, LSTM
    from keras.activations import sigmoid, tanh
    from keras.optimizers import Adam, Nadam, sgd
    from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
    from keras.layers.merge import concatenate

        
    #### model section ####
    vertex_input = Input(shape=(6,))
    x = Dense(units=9, activation='selu')(vertex_input)
    x = Dropout(0.2)(x)
    x = Dense(units=512, activation='selu')(x)
    x = Dense(units=256, activation='selu')(x)
    x = Dense(units=128, activation='selu')(x)
    x = Dense(units=64, activation='selu')(x)
    out = Dense(units=1, activation='selu')(x)

    out_trk = Dense(1, activation='sigmoid', name='out_trk')(out)

    tracks_input = Input(shape=(5,), name='tracks_input')
 
    x = concatenate([out,tracks_input],axis=-1)

    x = Dense(units=5,kernel_initializer='uniform',activation='selu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=256,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=128,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=64,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=32,kernel_initializer='uniform',activation='selu')(x)

    main_out = Dense(units=1,kernel_initializer='uniform',activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[vertex_input, tracks_input], outputs=[main_out, out_trk])

    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit([features_vtx,features_trk], [labels_vtx,labels_trk], epochs=epochs)
    
    ### PREDICTION  ###
    pred_labels = model.predict([features_vtx,features_trk])
    #probs = model.predict_proba([features_vtx,features_trk])
    #probs1D = probs.flatten()
    

    ### EVALUATION ###
    scores = model.evaluate([features_vtx,features_trk], [labels_vtx,labels_trk]) #verbose=0  
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))  
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))  
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))  
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))  

    return model, history, scores, pred_labels # probs1D ,probs


### PREDICTION METHOD ###
def prediction(model, x_test, y_test):
    
    result = model.predict(x_test)
    print(" ##AFTER## , PROBS shape and dimensions :", result.shape, result.ndim)
    probs = model.predict_proba(x_test)
    print(" ##AFTER## , PROBS shape and dimensions :", probs.shape, probs.ndim)
    probs1D = probs.flatten()
    print("1D_probs shape and dimensions :", probs1D.shape, probs1D.ndim)
    
    return result, probs, probs1D

### main method ###
def execute(nSamples, epochs, nEp, test_size, path):
    import InDetVtxClassPlot as plot
    import Utils as utl
    import os

    form = '.png'

    ## LOAD DATA ##
    feat_vtx, lab_vtx = load_data(ROOTFILE,DATASET_VTX,LAB_VTX, nSamples, 6)
    feat_trk, lab_trk = load_data(ROOTFILE,DATASET_TRK,LAB_TRK, nSamples, 5)
    print ("vertex features and label shape :", feat_vtx.shape, lab_vtx.shape)
    print ("tracks features and label shape :", feat_trk.shape, lab_trk.shape)
    
    ## LOAD MODEL ##
    vtx_mod = vertex_model()
    trk_mod = tracks_model()
    lstm_vtx_mod = LSTM_model()
    lstm_trk_mod = LSTM_model()
            
    ## VERTEX MODEL ##
    # print("########## FIRST MODEL ##########")
    # print("training & plotting vertices model")
    
    vrs = 'Roc_vertex_model'
    vhs = 'History_vertex_model'
    vps = 'Probs_vertex_model'
    vtx_roc_str = os.path.join(vrs+nEp+form)
    vtx_hst_str = os.path.join(vhs+nEp+form)
    vtx_prob_str = os.path.join(vps+nEp+form)

    vtx_history, vtx_scores, x_vtx_test, y_vtx_test, out_vtx_mod = train_model(vtx_mod, feat_vtx, lab_vtx, epochs)
    vtx_result, vtx_probs, vtx_probs1D = prediction(out_vtx_mod, x_vtx_test, y_vtx_test)
    plot.plot_roc(x_vtx_test, y_vtx_test, out_vtx_mod, vtx_roc_str, path)
    plot.plot_history(vtx_history, vtx_hst_str, path)
    plot.plot_probs(y_vtx_test, vtx_probs1D, vtx_prob_str, path)

    
    ## TRACKS MODEL ##
    print("########## SECOND MODEL ##########")
    print("training & plotting tracks model")

    trs = 'Roc_track_model'
    ths = 'History_track_model'
    tps = 'Probs_track_model'
    trk_roc_str = os.path.join(trs+nEp+form)
    trk_hst_str = os.path.join(ths+nEp+form)
    trk_prob_str = os.path.join(tps+nEp+form)

    trk_history, trk_scores, x_trk_test, y_trk_test, out_trk_mod = train_model(trk_mod, feat_trk, lab_trk, epochs)    
    trk_result, trk_probs, trk_probs1D = prediction(out_trk_mod, x_trk_test, y_trk_test)
    plot.plot_roc(x_trk_test, y_trk_test, out_trk_mod, trk_roc_str, path)
    plot.plot_history(trk_history, trk_hst_str, path)
    plot.plot_probs(y_trk_test, trk_probs1D, trk_prob_str, path)


    ## LSTM VERTEX METHOD ##
    print("########## THIRD MODEL ##########")
    print("training & plotting lSTM vertex model")

    #vlrs = 'Roc_vertex_lstm_model'
    vlhs = 'History_vertex_lstm_model'
    vlps = 'Prob_vertex_lstm_model'
    #vtx_lstm_roc_str = os.path.join(vlrs+nEp+form)
    vtx_lstm_hst_str = os.path.join(vlhs+nEp+form)
    vtx_lstm_prob_str = os.path.join(vlps+nEp+form)

    feat_vtx_3D, lab_vtx_3D = utl.twoD_to_3D(feat_vtx, lab_vtx, nSamples, 1, 6)
    lstm_vtx_hist, lstm_vtx_scores, lstm_vtx_x_test, lstm_vtx_y_test, lstm_vtx_mod = train_model(lstm_vtx_mod, feat_vtx_3D, lab_vtx_3D, epochs)
    lstm_vtx_result, lstm_vtx_probs, lstm_vtx_probs1D = prediction(lstm_vtx_mod, lstm_vtx_x_test, lstm_vtx_y_test)
    lstm_x_test_2D = utl.threeD_to2D(lstm_vtx_x_test, test_size, 6)
    lstm_y_test_2D = utl.threeD_to2D(lstm_vtx_y_test, test_size, 1)

    #plot.plot_roc(lstm_x_test_2D, lstm_y_test_2D, lstm_vtx_mod, vtx_lstm_roc_str, path)
    plot.plot_history(lstm_vtx_hist, vtx_lstm_hst_str, path)
    plot.plot_probs(lstm_y_test_2D, lstm_vtx_probs1D, vtx_lstm_prob_str, path)


    ## LSTM TRACKS METHOD ##
    print("########## FOURTH MODEL ##########")
    print("training & plotting lSTM tracks model")

    #tlrs = 'Roc_tracks_lstm_model'
    tlhs = 'History_tracks_lstm_model'
    tlps = 'Prob_tracks_lstm_model'
    #trk_lstm_roc_str = os.path.join(tlrs+nEp+form)
    trk_lstm_hst_str = os.path.join(tlhs+nEp+form)
    trk_lstm_prob_str = os.path.join(tlps+nEp+form)

    feat_trk_3D, lab_trk_3D = utl.twoD_to_3D(feat_trk, lab_trk, nSamples, 1, 5)
    lstm_trk_hist, lstm_trk_scores, lstm_trk_x_test, lstm_trk_y_test, lstm_trk_mod = train_model(lstm_trk_mod, feat_trk_3D, lab_trk_3D, epochs)
    lstm_trk_result, lstm_trk_probs, lstm_trk_probs1D = prediction(lstm_trk_mod, lstm_trk_x_test, lstm_trk_y_test)
    lstm_x_test_2D = utl.threeD_to2D(lstm_trk_x_test, test_size, 5)
    lstm_y_test_2D = utl.threeD_to2D(lstm_trk_y_test, test_size, 1)

    #plot.plot_roc(lstm_x_test_2D, lstm_y_test_2D, lstm_trk_mod, trk_lstm_roc_str, path)
    plot.plot_history(lstm_trk_hist, trk_lstm_hst_str, path)
    plot.plot_probs(lstm_y_test_2D, lstm_trk_probs1D, trk_lstm_prob_str, path)


    ## CONCATENATED METHOD ##
    print("########## FIFTH MODEL ##########")
    print("concat model")

    con_roc_vtx_str = 'Roc_vtx_concat_model'
    con_roc_trk_str = 'Roc_trk_concat_model'    
    chs = 'History_concat_model'
    con_prob_str = 'Probs_concat_model'
    con_hst_str = os.path.join(chs+nEp+form)
    
    out_con_mod, conc_history, conc_scores, pred, con_probs, con_probs1D = conc_method(feat_vtx, lab_vtx, feat_trk, lab_trk, 50)
    out_con_mod, conc_history, conc_scores, pred = conc_method(feat_vtx, lab_vtx, feat_trk, lab_trk, epochs)
    plot.plot_conc_history(conc_history, con_hst_str, path)

    plot.plot_roc( feat_vtx, lab_vtx, out_con_mod, con_roc_vtx_str)
    plot.plot_roc( feat_trk, lab_trk, out_con_mod, con_roc_trk_str)
    plot.plot_probs(lab_vtx, con_probs1D, vtx_prob_str, path)

    #from keras.utils import plot_model
    #plot_model(out_con_mod, to_file='conc_vtx_trk_model.png')
    

    return None



def main():

    ## test_size = 20% nSamples ##
    nSamples = 30000
    test_size = 6000
    epochs = 10
    epochs_name = '_100Epochs'
    path = '/home/atlas/sspinali/Qual_task/Plots/Plot_newCode/Test_13Mar/'

    execute(nSamples, epochs, epochs_name, test_size, path)

    return None


main()

    
