from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.activations import sigmoid, tanh
from keras.optimizers import Adam, Nadam, sgd
from keras.activations import softmax, sigmoid, relu, tanh, elu, selu
from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import uproot
import pandas
import numpy as np
import keras
import json
import os
import pickle


def load_data(path_to_load, data, lab, nSample, nFeat):
    print(" ...loading data... ")
    f = uproot.open(path_to_load)
    tree = f['Merged_Analysis;1']
    dataset = tree.pandas.df(data)
    features = dataset.loc[:, dataset.columns != lab].values
    labels = dataset.loc[:, lab].values
    features = np.resize(features, (nSample, nFeat))
    labels = np.resize(labels, (nSample,))
    return features, labels


def new_load_data(path_to_load, data, lab, nCol, nSample):
    f = uproot.open(path_to_load)
    tree = f['Merged_Analysis;1']
    dataset = tree.pandas.df(data)
    all_vtx = dataset.loc[lambda dataset: dataset[lab] == 0]
    merged = dataset.loc[lambda dataset: dataset[lab] == 1]
    print(" merged_vtx shape = ", merged.shape,
          " all_vtx shape = ", all_vtx.shape)
    dataset_mer = np.array(merged)
    dataset_all_vtx = np.resize(all_vtx, (len(dataset_mer), nCol))
    conc = np.concatenate((dataset_mer, dataset_all_vtx), axis=0)
    feat = np.delete(conc, np.s_[(nCol-1)], axis=1)
    lab = conc[:, (nCol-1)]
    print("BEFORE feat = ", feat.shape, " labels = ", lab.shape)
    features = np.resize(feat, (nSample, (nCol-1)))
    labels = np.resize(lab, (nSample,))
    print("AFTER feat = ", features.shape,
          " labels = ", labels.shape)
    return feat, lab, features, labels


def train_model(model, features, labels, epochs, epochs_str):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test))
    print(' ...Evaluating... ')
    scores = model.evaluate(X_test, y_test)
    print("SCORE = %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("SCORE = %s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    return history, scores, X_test, y_test, model


def prediction(model, x_test):
    print(" ...in prediction method... ")
    res = model.predict(x_test)
    res1D = res.flatten()
    res_roc = model.predict(x_test).ravel()
    probs = model.predict_proba(x_test)
    probs1D = probs.flatten()
    return res, res1D, res_roc, probs, probs1D


def double_input_prediction(model, x_test_1, x_test_2):
    print(" ...in double input prediction method... ")
    pred = model.predict([x_test_1, x_test_2])
    pred1D = pred.flatten()
    pred_roc = model.predict([x_test_1, x_test_2]).ravel()
    return pred, pred1D, pred_roc


def create_directories(path, method_dir):
    plot_dir = 'Plots'
    model_draft_dir = 'Model_draft'
    method_path = os.path.join(path, method_dir)
    plot_path = os.path.join(method_path, plot_dir)
    model_draft_path = os.path.join(method_path, model_draft_dir)
    if not os.path.exists(method_path):
        os.makedirs(method_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if not os.path.exists(model_draft_path):
        os.makedirs(model_draft_path)
    return plot_path, model_draft_path


def twoD_to_3D(features, labels, batch_size, timesteps, input_dim):
    features_3D = np.array(features, dtype=float)
    labels_3D = np.array(labels, dtype=float)
    features_3D.resize(batch_size, timesteps, input_dim)
    labels_3D.resize(batch_size, timesteps, 1)
    return features_3D, labels_3D


def threeD_to2D(vect, nSamples, nDim):
    vect = np.array(vect, dtype=int)
    vect.resize(nSamples, nDim)
    return vect
