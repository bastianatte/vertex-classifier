
def plot_roc(features, labels, model, string, path):
    import os
    import pandas
    from keras import layers, models
    from keras.models import Sequential
    import matplotlib.pyplot as plt
    from matplotlib import axes
    ######## PLOT ROC CURVE #########
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    print("... In plot ROC ... ")    
    
    y_pred_keras = model.predict(features).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt1 = plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    
    save_path = os.path.join(path,string)

    plt.savefig(save_path)
    plt.close()

    # pl4 = plt.figure(3)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1.05)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # pl4.savefig("/home/atlas/sspinali/Qual_task/Plots/plot_ROC_ZOOM_5_relu_Adam.png")
    # pl4.savefig("/home/atlas/sspinali/Qual_task/Plots/plot_ROC_ZOOM_5_relu_Adam.pdf")
    # pl4.savefig("/home/atlas/sspinali/Qual_task/Plots/plot_Roc_zoom_test1.png")
    # pl4.savefig("/home/atlas/sspinali/Qual_task/Plots/plot_ROC_zoom_test1.pdf")

    print("yuppyyy... ROC is done!!!")

    return None


def plot_history(hist, string, path):
    import pandas
    import matplotlib.pyplot as plt
    from matplotlib import axes
    import os

    print("... In plot histo ...")
    
    print("history keys : ",hist.history.keys())
    #  "Accuracy"

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

    plt.savefig(save_path)
    plt.close()

    print("yuppyyy... HIST is done!!!")

    return None


def plot_conc_history(hist, string, path):
    import os
    import pandas
    import matplotlib.pyplot as plt
    from matplotlib import axes
    
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
    #plt.ylim(-0.1, 1.1)
    plt.legend(['loss', 'main_output_loss', 'out_trk_loss', 'main_output_acc', 'out_trk_acc'], loc=5)

    save_path = os.path.join(path,string)
    plt.savefig(save_path)
    plt.close()

    print("yuppyyy...  HIST conc is done!!!")

    return None

def plot_probs(y_test, probsonedim, string, path):
    import os
    import matplotlib.pyplot as plt
    
    print("... In plot PROBS ...")

    plt.figure(1)

    back_grd = [prob for label, prob in zip(y_test,probsonedim) if label==0]
    target = [prob for label, prob in zip(y_test,probsonedim) if label==1]

    plt.hist(back_grd, alpha = 0.4, color='b')
    plt.hist(target, alpha = 0.4, color='r')
    plt.xlabel("prob")
    plt.ylabel("# vertices")
    plt.title("probs charts")
    plt.xlim(-0.2, 1.2)

    save_path = os.path.join(path,string)
    plt.savefig(save_path)
    plt.close()

    print("yupyyy...  PROBS is done!!! ")
    
    return None
 
def plot_feat_imp(features, labels, X_test, X_train):
    import pandas
    import matplotlib.pyplot as plt
    from matplotlib import axes

    print("In plot feat. importance")
    #from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from pylab import *
    
    rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf_fit = rnd_clf.fit(features, labels)
    
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)
    
    #figure, axes = plt.subplots()
    axes= figure().add_subplot(111)
    
    plt.title('Feature Importances')
    axes.barh(range(len(indices)), importances[indices], color='b', align='center')
    #y_labels = axes.get_yticks().tolist()
    #y_labels[0] = ""
    #y_labels[1] = ""
    #y_labels[3] = ""
    #y_labels[5] = ""
    #y_labels[7] = ""
    #y_labels[9] = ""
    #y_labels[2],y_labels[4],y_labels[6], y_labels[8] = "z_pos","trks","spt2","chsq"
    #axes.set_yticklabels(y_labels)
    #plt.xlabel('Relative Importance')
    plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_5_relu_Adam.pdf")
    plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_5_relu_Adam.png")
    #plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_test1.pdf")
    #plt.savefig("/home/atlas/sspinali/Qual_task/Plots/feat_import_test1.png")

    #### Random Forest Score ####
    print("Random Forest Score = ", rnd_clf.score(X_test, y_test))

    #### Random Forest Feature Importance ####
    feature_importances = pandas.DataFrame(rnd_clf.feature_importances_)
    print(feature_importances)
    importances = rnd_clf.feature_importances_
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    print("Plot feat. importance done")

    return None



