
def vertex_model(in_unit,loss):   
    ### keras ###                                                                                                                     
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.activations import sigmoid, tanh
    from keras.optimizers import Adam, Nadam, sgd
    from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, sparse_categorical_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity

    
    model = Sequential()
    
    model.add(Dense(units=in_unit,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    
    model.add(Dropout(0.2))
    
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

    model.add(Dense(units=32,
                    kernel_initializer='uniform',
                    activation='selu'))
    
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))
    
    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    ### try o use cross entropy loss https://keras.io/losses/

    return model


### tracks model ###
def tracks_model(nFeat):
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
    
    model.add(Dense(units=nFeat,
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

def LSTM_model(nFeat):
    ### keras ###
    import keras
    from keras import layers, models
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from keras.activations import sigmoid
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    
    model = Sequential()

    model.add(LSTM( 1024, input_shape=(1,nFeat), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM( 1024, input_shape=(1,nFeat), return_sequences=True))
    model.add(LSTM( 256, input_shape=(1,nFeat), return_sequences=True))
    
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
 
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])

    return model


def conc_model(nVtxFeat, nTrkFeat, features_vtx, labels_vtx, features_trk, labels_trk, epochs):
    
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
    vertex_input = Input(shape=(nVtxFeat,))
    x = Dense(units=9, activation='selu')(vertex_input)
    x = Dropout(0.2)(x)
    x = Dense(units=512, activation='selu')(x)
    x = Dense(units=256, activation='selu')(x)
    x = Dense(units=128, activation='selu')(x)
    x = Dense(units=64, activation='selu')(x)
    out = Dense(units=1, activation='selu')(x)

    out_trk = Dense(1, activation='sigmoid', name='out_trk')(out)

    tracks_input = Input(shape=(nTrkFeat,), name='tracks_input')
 
    x = concatenate([out,tracks_input],axis=-1)

    x = Dense(units=5,kernel_initializer='uniform',activation='selu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=256,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=128,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=64,kernel_initializer='uniform',activation='selu')(x)
    x = Dense(units=32,kernel_initializer='uniform',activation='selu')(x)

    main_out = Dense(units=1,kernel_initializer='uniform',activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[vertex_input, tracks_input], 
                  outputs=[main_out, out_trk])

    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit([features_vtx,features_trk], [labels_vtx,labels_trk], epochs=epochs)
    
    ### PREDICTION  ###
    pred_labels = model.predict([features_vtx,features_trk])
    
    ### EVALUATION ###
    scores = model.evaluate([features_vtx,features_trk], [labels_vtx,labels_trk]) #verbose=0  
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))  
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))  
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))  
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))  

    return model, history, scores, pred_labels # probs1D ,probs

    
def add_model(nVtxFeat, nTrkFeat, feat_vtx, lab_vtx, feat_trk, lab_trk, ep):
    print(" ... Add() model... ")
    import keras
    from keras.optimizers import Adam
    from sklearn.model_selection import train_test_split

    #### input layers ####
    vtx_input = keras.layers.Input(shape=(1,nVtxFeat,))
    trk_input = keras.layers.Input(shape=(1,nTrkFeat,))

    #### NET 1 ####
    x1 = keras.layers.Dense(nVtxFeat, activation='relu')(vtx_input)    
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Dense(256, activation='relu')(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    x1 = keras.layers.Dense(32, activation='relu')(x1)
    x1 = keras.layers.LSTM(12)(x1)

    #### NET 2 ####
    x2 = keras.layers.Dense(nTrkFeat, activation='relu')(trk_input)
    x2 = keras.layers.LSTM(128, return_sequences=True)(x2)
    x2 = keras.layers.LSTM(64, return_sequences=True)(x2)
    #x2 = keras.layers.LSTM(32, input_shape=(1,5), return_sequences=False)(x2)
    x2 = keras.layers.LSTM(12)(x2)
    
    #### ADDED BOTH NET ###    
    # equivalent to added = keras.layers.add([x1, x2])
    added = keras.layers.Add()([x1, x2])
    out = keras.layers.Dense(1)(added)
    
    ### MODEL DEFINITION ###
    model = keras.models.Model(inputs=[vtx_input, trk_input], outputs=out)
    
    optimizer = Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)                                                               
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    X_train_vtx, X_test_vtx, y_train_vtx, y_test_vtx = train_test_split(feat_vtx, 
                                                                        lab_vtx, 
                                                                        test_size=0.2, 
                                                                        random_state=42)

    X_train_trk, X_test_trk, y_train_trk, y_test_trk = train_test_split(feat_trk, 
                                                                        lab_trk, 
                                                                        test_size=0.2, 
                                                                        random_state=42)


    ### FITTING STEP ###
    hist = model.fit([X_train_vtx, X_train_trk],
                     y_train_vtx,
                     validation_split=0.2,
                     batch_size=64,
                     epochs=ep)

    ### EVALUATION ###
    scores = model.evaluate([X_train_vtx, X_train_trk], 
                            y_train_vtx) #verbose=0 
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))  
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  
    


    return model, hist, scores, X_test_vtx, y_test_vtx, X_test_trk, y_test_trk   


def new_concat_model(feat_vtx, lab_vtx, feat_trk_3D, lab_trk_3D, ep, nVtxFeat, nTrkFeat, loss):
    print(" ... concatenate() model... ")
    import keras
    from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, sparse_categorical_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity
    from sklearn.model_selection import train_test_split
        
    input1 = keras.layers.Input(shape=(nVtxFeat,))
    x1 = keras.layers.Dense(512, activation='selu')(input1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Dense(256, activation='selu')(x1)    
    x1 = keras.layers.Dense(128, activation='selu')(x1)    #128
    #x1 = keras.layers.Dense(512, activation='selu')(x1)     #64
    x1 = keras.layers.Dense(64, activation='selu')(x1)     #32
    #x1 = keras.layers.Dense(128, activation='selu')(x1)     #16
    x1 = keras.layers.Dense(32, activation='selu')(x1)     
    #x1 = keras.layers.Dense(32, activation='selu')(x1)     
    x1 = keras.layers.Dense(16, activation='selu')(x1)     
 
    input2 = keras.layers.Input(shape=(1,nTrkFeat,))
    x2 = keras.layers.LSTM(512, return_sequences=True)(input2)
    x2 = keras.layers.Dropout(0.2)(x2)
    x2 = keras.layers.LSTM(256, return_sequences=True)(x2)
    x2 = keras.layers.LSTM(128, return_sequences=True)(x2)
    #x2 = keras.layers.LSTM(512, return_sequences=True)(x2)
    x2 = keras.layers.LSTM(64, return_sequences=True)(x2)
    #x2 = keras.layers.LSTM(128, return_sequences=True)(x2)
    x2 = keras.layers.LSTM(32, return_sequences=True)(x2)
    #x2 = keras.layers.LSTM(32, return_sequences=True)(x2)
    x2 = keras.layers.LSTM(16, return_sequences=False)(x2)
    
    merge = keras.layers.concatenate([x1,x2], axis=-1)

    out = keras.layers.Dense(1, activation='sigmoid') (merge)
    #out = keras.layers.Dense(2, activation='sigmoid') (merge)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    optimizer = keras.optimizers.Adam(lr=0.001) #decay=1e-6, momentum=0.9, nesterov=True)
    
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss=loss, #loss=mean_absolute_error, #loss=mean_absolute_percentage_error, #loss='mean_squared_error',
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    X_train_vtx, X_test_vtx, y_train_vtx, y_test_vtx = train_test_split(feat_vtx,
                                                                        lab_vtx,
                                                                        test_size=0.2,
                                                                        random_state=42)

    X_train_trk, X_test_trk, y_train_trk, y_test_trk = train_test_split(feat_trk_3D,
                                                                        lab_trk_3D,
                                                                        test_size=0.2,
                                                                        random_state=42)


    hist = model.fit([X_train_vtx, X_train_trk],
                     y_train_vtx,
                     validation_split=0.2,
                     batch_size=64,
                     epochs=ep)

    ### EVALUATION ###
    scores = model.evaluate([X_test_vtx, X_test_trk],
                            y_test_vtx) #verbose=0
    
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    return model, hist, scores, X_test_vtx, y_test_vtx, X_test_trk, y_test_trk
