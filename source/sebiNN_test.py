{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import uproot "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas                                                                      \n",
    "import keras                                                                       \n",
    "                                                                                   \n",
    "from keras import layers, models                                                   \n",
    "from keras.models import Sequential                                                \n",
    "from keras.layers import Dense  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "f = uproot.open(\"/home/sebi/Documents/lxplus-stuff/QT_NN/Data/VertexNN_uproot.root\")                                                                                 \n",
    "tree = f['Merged_Analysis;1']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        Match_tracks  Merge_tracks  Split_tracks  Fake_tracks        ismerge\n",
      "count  240186.000000  58296.000000   1221.000000  3972.000000  303675.000000\n",
      "mean       23.820311     40.358669      9.360360     3.752266       0.191968\n",
      "std        24.439472     37.786396     18.408119     4.208516       0.393849\n",
      "min         2.000000      2.000000      2.000000     2.000000       0.000000\n",
      "25%         6.000000     10.000000      3.000000     2.000000       0.000000\n",
      "50%        15.000000     27.000000      4.000000     3.000000       0.000000\n",
      "75%        33.000000     61.000000      7.000000     4.000000       0.000000\n",
      "max       230.000000    285.000000    189.000000   124.000000       1.000000\n"
     ]
    }
   ],
   "source": [
    "##################DATAFRAME##################\n",
    "dataframe = tree.pandas.df()                                                       \n",
    "\n",
    "dataset = tree.pandas.df([\"Match_tracks\",\"Merge_tracks\",\"Split_tracks\",\"Fake_tracks\",\"ismerge\"])                                                                     \n",
    "print (dataset.describe())                                                         \n",
    "                                                                                   \n",
    "                                                                                   \n",
    "features = dataset.loc[:, dataset.columns != \"ismerge\"].values   \n",
    "labels = dataset.loc[:, \"ismerge\"].values                                       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_23 (Dense)             (None, 64)                320       \n",
      "_________________________________________________________________\n",
      "dense_24 (Dense)             (None, 8)                 520       \n",
      "_________________________________________________________________\n",
      "dense_25 (Dense)             (None, 1)                 9         \n",
      "=================================================================\n",
      "Total params: 849\n",
      "Trainable params: 849\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Dense(units=64, activation='relu', input_dim=4))\n",
    "model.add(Dense(units=8, activation='softmax'))\n",
    "model.add(Dense(units=1, activation='softmax'))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='binary_crossentropy',\n",
    "              optimizer='sgd',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 303371 samples, validate on 304 samples\n",
      "Epoch 1/3\n",
      "303371/303371 [==============================] - 10s 32us/step - loss: nan - acc: 0.0000e+00 - val_loss: nan - val_acc: 0.0000e+00\n",
      "Epoch 2/3\n",
      "303371/303371 [==============================] - 10s 32us/step - loss: nan - acc: 0.0000e+00 - val_loss: nan - val_acc: 0.0000e+00\n",
      "Epoch 3/3\n",
      "303371/303371 [==============================] - 10s 32us/step - loss: nan - acc: 0.0000e+00 - val_loss: nan - val_acc: 0.0000e+00\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f34aad0a7b8>"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(features, labels, epochs=3, batch_size=32,validation_split=0.001,shuffle=True) #verbose=0 \n",
    "#model.fit(features, labels, epochs=3, batch_size=100) #verbose=0                   \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "303675/303675 [==============================] - 5s 15us/step\n",
      "acc: 0.00%\n"
     ]
    }
   ],
   "source": [
    "scores = model.evaluate(features, labels) #verbose=0  \n",
    "print(\"%s: %.2f%%\" % (model.metrics_names[1], scores[1]*100))                      \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
