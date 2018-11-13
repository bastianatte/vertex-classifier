import uproot
import pandas
import matplotlib.pyplot as plt

#f = uproot.open("/home/sebi/Documents/lxplus-stuff/QT_NN/Data//NN_Test_main.root")
f = uproot.open("/home/sebi/Documents/lxplus-stuff/QT_NN/Data//NN_full.root")
tree = f['Merged_Analysis;1']
#tree1 = f['Merged_Analysis']

tree.keys()
print(tree.keys())
#print(tree1.keys())

dataframe = tree.pandas.df()
#print (dataframe)

dataset = tree.pandas.df(["Match_z","Match_tracks","Merge_z","Merge_tracks","Split_z","Split_tracks","Fake_z","Fake_tracks","ismerge"])
dat = tree.pandas.df(["Split_z","Split_tracks","Fake_z","Fake_tracks","ismerge"])

diff = tree.pandas.df(["Match_z","Merge_z"])
diff = diff.cumsum()
plt.figure()
diff.plot()


#print (dataset.shape)
#print(dataset)
#print(dat)
#print (dataset.dtypes)
#print (dataset.head())
#print (dataset.tail())
#print (dataset.index)
#print (dataset.describe())



