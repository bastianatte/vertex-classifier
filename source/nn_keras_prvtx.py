import uproot
import pandas
import keras

from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense

##############  LOAD DATA  ##############

#f = uproot.open("/home/sebi/Documents/lxplus-stuff/QT_NN/Data/NN_Test_main.root")
#f = uproot.open("/home/sebi/Documents/lxplus-stuff/QT_NN/Data/NN_pos_track.root")
f = uproot.open("/home/sebi/Documents/lxplus-stuff/QT_NN/Data/NN_full.root")

tree = f['Merged_Analysis;1']


#############  DATAFRAME ############

dataframe = tree.pandas.df()

#dataset = tree.pandas.df(["Match_z","Merge_z","ismerge"])
#dataset = tree.pandas.df(["ismatch_z","Match_tracks","Merge_tracks","ismerge_z","ismerge"])
dataset = tree.pandas.df(["Match_z","Match_tracks","Merge_z","Merge_tracks","Split_z","Split_tracks","Fake_z","Fake_tracks","ismerge"])
print(dataset)

features = dataset.loc[:, dataset.columns != "ismerge"].values
#print(features)
print("The shape of the features is = ", features.shape)

labels = dataset.loc[:, "ismerge"].values
#print(labels)
print("The shape of the label is = ", labels.shape)


############ DEFINE MODEL  ##########

model = Sequential()

#model.add(Dense(12, input_dim=2, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(12, input_dim=4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.summary()

########### Compile model  ############                                                                                         

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

########## Fit the model  ###########                                                                                          

model.fit(features, labels, epochs=5, batch_size=50) #verbose=0

######### evaluate the model  ###########                                                                                     

scores = model.evaluate(features, labels) #verbose=0                                                     
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

############ prediction ###############

predict = model.predict(features)
print (predict, labels)
