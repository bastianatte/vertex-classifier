from keras import layers, models
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib import axes
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from matplotlib import axes
import os
import pandas
import numpy as np
from pylab import *



def plot_roc(features, labels, y_pred_keras, string, path):
    """
    This method need as parameters features, and y_pred_keras 
    for plotting the point, and string, path for the repo. 
    """

    print("... In plot ROC ... ")    
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt1 = plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    save_path = os.path.join(path,string)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("ROC is done!!!")
    return None


def plot_zoom_roc(features, labels, y_pred_keras, string, path):
    """
    This method need as parameters features, and y_pred_keras 
    for plotting the point, and string, path for the repo. 
`   """

    print("... In plot ROC ... ")    
    pl4 = plt.figure(3)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1.05)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    save_path = os.path.join(path,string)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("zoom ROC is done!!!")
    return None


def plot_history(hist, string, path):
    print("... In plot histo ...")    
    print("history keys : ",hist.history.keys())
    plt2 = plt.figure(2)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model acc and loss')
    plt.ylabel('accuracy & loss % ')
    plt.xlabel('epoch')
    plt.ylim(0, 1)
    plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc=5)
    save_path = os.path.join(path,string)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("yuppyyy... HIST is done!!!")
    return None


def plot_conc_history(hist, string, path):
    print("... In plot conc history ...")
    print("history keys : ",hist.history.keys())
    plt3 = plt.figure(3)
    plt.plot(hist.history['main_output_acc'])
    plt.plot(hist.history['out_trk_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['main_output_loss'])
    plt.plot(hist.history['out_trk_loss'])
    plt.title('Concatenate model')
    plt.ylabel('accuracy & loss %')
    plt.xlabel('epoch')
    plt.legend(['loss', 'main_output_loss', 'out_trk_loss', 'main_output_acc', 'out_trk_acc'], loc=5)
    save_path = os.path.join(path,string)
    plt.savefig(save_path)
    plt.close()
    print("yuppyyy...  HIST conc is done!!!")
    return None


def plot_probs(y_test, probsonedim, string, path):
    print("... In plot PROBS ...")
    plt.figure(1)
    all_vtx = [prob for label, prob in zip(y_test,probsonedim) if label==0]
    merged_vtx = [prob for label, prob in zip(y_test,probsonedim) if label==1]
    plt.hist(all_vtx, alpha = 0.4, bins=200, color='b')
    plt.hist(merged_vtx, alpha = 0.4, bins=200, color='r')
    plt.xlabel("prob")
    plt.ylabel("# vertices")
    plt.title("probs charts")
    plt.xlim(-0.1, 1.1)
    plt.legend(['all_vtx', 'merged_vtx'], loc=1)
    save_path = os.path.join(path,string)
    plt.savefig(save_path, dpi=200)
    plt.close()
    mer_cnt_ye s= 0
    mer_cnt_no = 0
    for i, value_mer in enumerate(merged_vtx):
        i += i
        if value_mer > 0.7:
            mer_cnt_ye += 1
        if value_mer < 0.3:
            mer_cnt_no += 1
    print('found', i, 'merged vertices!!! ', mer_cnt_ye, 'with probability higher than 0.7, and ', mer_cnt_no , 'with probability lower than 0.3')
    nomer_cnt_ye = 0
    nomer_cnt_no = 0
    for j, value_nomer in enumerate(all_vtx):
        j += j
        if value_nomer > 0.7:
            nomer_cnt_ye += 1
        if value_nomer < 0.3:
            nomer_cnt_no += 1
    print('found', j, 'not merged vertices!!! ', nomer_cnt_ye, 'with probability higher than 0.7, and ', nomer_cnt_no , 'with probability lower than 0.3')
    print("yupyyy...  PROBS is done!!! ")
    return None
 

def plot_feat_imp(features, labels, X_test, X_train):    
    rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf_fit = rnd_clf.fit(features, labels)
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)
    axes= figure().add_subplot(111)
    plt.title('Feature Importances')
    axes.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_5_relu_Adam.pdf")
    plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_5_relu_Adam.png")
    print("Random Forest Score = ", rnd_clf.score(X_test, y_test))
    feature_importances = pandas.DataFrame(rnd_clf.feature_importances_)
    print(feature_importances)
    importances = rnd_clf.feature_importances_
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))    
    print("Plot feat. importance done")
    return None



