import numpy as np
from keras.utils import plot_model
import InDetVtxClassPlot as plot
import InDetVtxModels as mod
import Utils as utl
import os
import json
import pickle
import sys
import time
import os
from keras.losses import logcosh, binary_crossentropy, mean_squared_error,\
    mean_absolute_error


def execute(nSamples,
            nVtxFeats,
            nTrackFeats,
            epochs,
            loss,
            nEp,
            test_size,
            path,
            m_name,
            models):
    trk_nSamples = 2000000
    print("vertex Samples = ", nSamples, "tracks Samples = ", trk_nSamples)
    trk_test_size = 500000
    DATASET_VTX = ["z_pos", "trks", "clos_tp", "sumpt",
                   "W1", "W2", "w4", "w4_d0", "ch_dof", "isMerge"]
    LAB_VTX = "isMerge"
    DATASET_TRK = ["pt", "eta", "phi", "d0", "z0", "isMerge_trk_2"]
    LAB_TRK = "isMerge_trk_2"
    ROOTFILE = '/home/spinali/lapzhd/Qual_task/Data/50K_Merged_Vtx_sample.root'
    mod_rep = 'Models'
    png = '.png'
    json = '.json'
    h5 = '.h5'
    dat = '.dat'
    print(" ...load vtx data... ")
    feat_vtx_noResh, lab_vtx_noResh, feat_vtx, lab_vtx = utl.new_load_data(ROOTFILE,
                                                                           DATASET_VTX,
                                                                           LAB_VTX,
                                                                           nVtxFeats+1,
                                                                           nSamples)
    print(" ...load trk data... ")
    feat_trk_noResh, lab_trk_noResh, feat_trk, lab_trk = utl.new_load_data(ROOTFILE,
                                                                           DATASET_TRK,
                                                                           LAB_TRK,
                                                                           nTrackFeats+1,
                                                                           trk_nSamples)

    if (models["vertex"] == 1):
        print("########## FIRST MODEL ##########")
        print(" ...training vertex model... ")
        method_dir = 'Vertex_Method'
        plot_path, model_path = utl.create_directories(path, method_dir)
        vrs = 'Roc_vertex_model'
        vhs = 'History_vertex_model'
        vps = 'Probs_vertex_model_predict_proba'
        vps_1 = 'Probs_vertex_model_predict'
        vtx_roc_str = os.path.join(vrs + nEp + png)
        vtx_hst_str = os.path.join(vhs + nEp + png)
        vtx_prob_str = os.path.join(vps + nEp + png)
        vtx_prob_str_1 = os.path.join(vps_1 + nEp + png)
        vtx_mod = mod.vertex_model(nVtxFeats, loss)
        vtx_history, vtx_scores, x_vtx_test, y_vtx_test, out_vtx_mod = utl.train_model(
            vtx_mod, feat_vtx, lab_vtx, epochs, nEp)
        vtx_result, vtx_result1D, vtx_result_roc, vtx_probs, vtx_probs1D = utl.prediction(
            out_vtx_mod, x_vtx_test)
        print(" ...plotting vertex model... ")
        plot.plot_roc(x_vtx_test, y_vtx_test, out_vtx_mod,
                      vtx_result_roc, vtx_roc_str, plot_path)
        plot.plot_history(vtx_history, vtx_hst_str, plot_path)
        plot.plot_probs(y_vtx_test, vtx_probs1D, vtx_prob_str, plot_path)
        plot.plot_probs(y_vtx_test, vtx_result1D, vtx_prob_str_1, plot_path)
        plot_model(out_vtx_mod, to_file=model_path +
                   '/vertex_model.png', show_shapes=True)
        print(" ...vertex model... is DONE!!! ")
    if (models["tracks"] == 1):
        print("########## SECOND MODEL ##########")
        print("training tracks model")
        method_dir = 'Tracks_Method'
        plot_path, model_path = utl.create_directories(path, method_dir)
        trs = 'Roc_track_model'
        ths = 'History_track_model'
        tps = 'Probs_track_model'
        trk_roc_str = os.path.join(trs + nEp + png)
        trk_hst_str = os.path.join(ths + nEp + png)
        trk_prob_str = os.path.join(tps + nEp + png)
        trk_mod = mod.tracks_model(nTrackFeats)
        trk_history, trk_scores, x_trk_test, y_trk_test, out_trk_mod = utl.train_model(
            trk_mod, feat_trk, lab_trk, epochs, nEp)
        trk_result, trk_result1D, trk_result_roc, trk_probs, trk_probs1D = utl.prediction(
            out_trk_mod, x_trk_test)  # questo e' quello nuovo modificato
        # trk_result, trk_probs, trk_probs1D = utl.prediction(out_trk_mod, x_trk_test)
        print(" ...plotting tracks model... ")
        plot.plot_roc(x_trk_test, y_trk_test, out_trk_mod,
                      trk_result_roc, trk_roc_str, plot_path)
        plot.plot_history(trk_history, trk_hst_str, plot_path)
        plot.plot_probs(y_trk_test, trk_probs1D, trk_prob_str, plot_path)
        plot_model(out_vtx_mod, to_file=model_path +
                   '/Tracks_model.png', show_shapes=True)

        print(" ...tracks model... is DONE!!! ")

    if (models["vertex_lstm"] == 1):
        print("########## THIRD MODEL ##########")
        print("training & plotting lSTM vertex model")
        method_dir = 'Vertex_LSTM_Method'
        plot_path, model_path = utl.create_directories(path, method_dir)
        vlrs = 'Roc_vertex_lstm_model'
        vlhs = 'History_vertex_lstm_model'
        vlps = 'Prob_vertex_lstm_model'
        vtx_lstm_roc_str = os.path.join(vlrs + nEp + png)
        vtx_lstm_hst_str = os.path.join(vlhs + nEp + png)
        vtx_lstm_prob_str = os.path.join(vlps + nEp + png)
        lstm_vtx_mod = mod.LSTM_model(nTrackFeats)
        feat_vtx_3D, lab_vtx_3D = utl.twoD_to_3D(
            feat_vtx, lab_vtx, nSamples, 1, 6)
        lstm_vtx_hist, lstm_vtx_scores, lstm_vtx_x_test, lstm_vtx_y_test, lstm_vtx_mod = utl.train_model(
            lstm_vtx_mod, feat_vtx_3D, lab_vtx_3D, epochs, nEp)
        lstm_vtx_result, lstm_vtx_result1D, lstm_vtx_result_roc, lstm_vtx_probs, lstm_vtx_probs1D = utl.prediction(
            lstm_vtx_mod, lstm_vtx_x_test)  # modificato, controllare se funziona!!!!
        lstm_x_test_2D = utl.threeD_to2D(lstm_vtx_x_test, test_size, 6)
        lstm_y_test_2D = utl.threeD_to2D(lstm_vtx_y_test, test_size, 1)

        ##### saving model and infos #######
        # wei_name = 'weight_test_800'
        # mod_name = 'arc_test_800'
        # wei = os.path.join(path,mod_rep,wei_name + h5)
        # arch = os.path.join(path,mod_rep,mod_name + json)
        # Save model
        # lstm_vtx_mod.save('/home/atlas/sspinali/Qual_task/Plots/Plot_newCode/Test_22Mar/Models/lstm_vtx_model_test_800.h5')
        # Save the weights
        # lstm_vtx_mod.save_weights(wei)
        # Save the model architecture
        # with open(arch, 'w') as file:
        #     file.write(lstm_vtx_mod.to_json())

        #### PLOTS #####
        plot.plot_roc(lstm_x_test_2D, lstm_y_test_2D, lstm_vtx_mod,
                      lstm_vtx_result_roc, vtx_lstm_roc_str, plot_path)
        plot.plot_history(lstm_vtx_hist, vtx_lstm_hst_str, plot_path)
        plot.plot_probs(lstm_y_test_2D, lstm_vtx_probs1D,
                        vtx_lstm_prob_str, plot_path)
        plot_model(lstm_vtx_mod, to_file=model_path +
                   '/Tracks_model.png', show_shapes=True)

    ################################ LSTM TRACKS METHOD ##########################################

    if (models["tracks_lstm"] == 1):
        print("########## FOURTH MODEL ##########")
        print("training & plotting lSTM tracks model")

        ### DIRS ###
        method_dir = 'Tracks_LSTM_Method'
        plot_path, model_path = utl.create_directories(path, method_dir)

        ### FILES ###
        tlrs = 'Roc_tracks_lstm_model'
        tlhs = 'History_tracks_lstm_model'
        tlps = 'Prob_tracks_lstm_model'
        trk_lstm_roc_str = os.path.join(tlrs + nEp + png)
        trk_lstm_hst_str = os.path.join(tlhs + nEp + png)
        trk_lstm_prob_str = os.path.join(tlps + nEp + png)

        ### MODEL ###
        lstm_trk_mod = mod.LSTM_model(5)
        feat_trk_3D, lab_trk_3D = utl.twoD_to_3D(
            feat_trk, lab_trk, trk_nSamples, 1, 5)
        lstm_trk_hist, lstm_trk_scores, lstm_trk_x_test, lstm_trk_y_test, lstm_trk_mod = utl.train_model(
            lstm_trk_mod, feat_trk_3D, lab_trk_3D, epochs, nEp)

        ### PREDICT ###
        lstm_trk_result, lstm_trk_result1D, lstm_trk_result_roc, lstm_trk_probs, lstm_trk_probs1D = utl.prediction(
            lstm_trk_mod, lstm_trk_x_test)
        #add_pred, add_pred1D, add_pred_roc = utl.double_input_prediction(add_model, add_x_test_vtx, add_x_test_trk)

        lstm_x_test_2D = utl.threeD_to2D(lstm_trk_x_test, trk_test_size, 5)
        lstm_y_test_2D = utl.threeD_to2D(lstm_trk_y_test, trk_test_size, 1)

        ### PLOTS ###
        plot.plot_roc(lstm_x_test_2D, lstm_y_test_2D, lstm_trk_mod,
                      lstm_trk_result_roc, trk_lstm_roc_str, plot_path)
        plot.plot_history(lstm_trk_hist, trk_lstm_hst_str, plot_path)
        plot.plot_probs(lstm_y_test_2D, lstm_trk_probs1D,
                        trk_lstm_prob_str, plot_path)
        plot_model(lstm_trk_mod, to_file=model_path +
                   '/Tracks_lstm_model.png', show_shapes=True)

    #################################### CONCATENATED METHOD #####################################

    if (models["conc"] == 1):
        print("########## FIFTH MODEL ##########")
        print("concat model")

        con_roc_vtx_str = 'Roc_vtx_concat_model'
        con_roc_trk_str = 'Roc_trk_concat_model'
        chs = 'History_concat_model'
        con_prob_str = 'Probs_concat_model'
        con_hst_str = os.path.join(chs+nEp+png)

        #out_con_mod, conc_history, conc_scores, pred, con_probs, con_probs1D = conc_model(feat_vtx, lab_vtx, feat_trk, lab_trk, epochs)
        #out_con_mod, conc_history, conc_scores, pred = conc_model(feat_vtx, lab_vtx, feat_trk, lab_trk, epochs)

        out_con_mod, conc_history, conc_scores, pred = mod.conc_model(
            nVtxFeats, nTrackFeats, feat_vtx, lab_vtx, feat_trk, lab_trk, epochs)
        con_probs, con_probs1D, con_probs_roc = utl.double_input_prediction(
            add_model, add_x_test_vtx, add_x_test_trk)  # need to be checked before using it

        plot.plot_conc_history(conc_history, con_hst_str, path)
        plot.plot_roc(feat_vtx, lab_vtx, out_con_mod,
                      con_probs_roc, con_roc_vtx_str, path)
        plot.plot_roc(feat_trk, lab_trk, out_con_mod, con_roc_trk_str, path)
        plot.plot_probs(lab_vtx, con_probs1D, vtx_prob_str, path)

        plot_model(out_con_mod, to_file='conc_vtx_trk_model.png')

    ##################################### ADD METHOD ###############################################

    if (models["add"] == 1):
        print("########## SIXTH MODEL ##########")

        ### DIRS ###
        method_dir = 'Add_Model'
        plot_path, model_path = utl.create_directories(path, method_dir)

        ### FILES NAME ###
        tars = 'Roc_added_model'
        tahs = 'History_added_model'
        taps = 'Prob_added_model'
        mod_name = 'added_model'

        added_roc_str = os.path.join(tars + nEp + png)
        added_hst_str = os.path.join(tahs + nEp + png)
        added_prob_str = os.path.join(taps + nEp + png)
        mod_str = os.path.join(mod_name + png)

        ### from 2D to 3D ###
        feat_vtx_add_3D, lab_vtx_add_3D = utl.twoD_to_3D(
            feat_vtx, lab_vtx, nSamples, 1, nVtxFeats, nTrackFeats)
        feat_trk_add_3D, lab_trk_add_3D = utl.twoD_to_3D(
            feat_trk, lab_trk, nSamples, 1, 5)

        ### model ###
        add_model, add_hist, add_scores, add_x_test_vtx, add_y_test_vtx, add_x_test_trk, add_y_test_trk = mod.add_model(
            nVtxFeats, nTrackFeats, feat_vtx_add_3D, lab_vtx, feat_trk_add_3D, lab_trk_add_3D, epochs)
        add_pred, add_pred1D, add_pred_roc = utl.double_input_prediction(
            add_model, add_x_test_vtx, add_x_test_trk)

        ### from 3D to 2D ###
        add_y_test_vtx = np.array(add_y_test_vtx, dtype=float)
        add_y_test_vtx.resize(test_size, 1, 1)
        added_x_test_2D = utl.threeD_to2D(add_x_test_vtx, test_size, nVtxFeats)
        added_y_test_2D = utl.threeD_to2D(
            add_y_test_vtx, test_size, nTrackFeats)

        print(" ...new added model plotting ... ")

        plot.plot_roc(added_x_test_2D, added_y_test_2D, add_model,
                      add_pred_roc, added_roc_str, plot_path)
        plot.plot_history(add_hist, added_hst_str, plot_path)
        plot.plot_probs(added_y_test_2D, add_pred1D, added_prob_str, plot_path)
        plot_model(add_model, to_file=model_path +
                   '/add_model.png', show_shapes=True)

        print(" ...add() model...  DONE!!!!!!  ")

    ################################## NEW CONCATENATED MODEL #######################################

    if (models["conc_best"] == 1):
        print("########## SEVENTH MODEL ##########")
        ### DIRS ###
        method_dir = 'new_concatenate_model'
        plot_path, model_path = utl.create_directories(path, method_dir)

        ### FILES NAME ###
        tars_1 = 'Roc_new_conc_model'
        tahs_1 = 'History_new_conc_model'
        taps_1 = 'Prob_new_conc_model'
        mod_name_1 = 'new_conc_model'
        new_conc_roc_str = os.path.join(tars_1 + nEp + png)
        new_conc_hst_str = os.path.join(tahs_1 + nEp + png)
        new_conc_prob_str = os.path.join(taps_1 + nEp + png)
        mod_str = os.path.join(mod_name_1 + m_name + png)

        ### MODEL ###
        feat_trk_new_conc_3D, lab_trk_new_conc_3D = utl.twoD_to_3D(
            feat_trk, lab_trk, nSamples, 1, nTrackFeats)
        start = time.time()
        new_conc_model, new_conc_hist, new_conc_scores, new_conc_x_test_vtx, new_conc_y_test_vtx, new_conc_x_test_trk, new_conc_y_test_trk = mod.new_concat_model(
            feat_vtx, lab_vtx, feat_trk_new_conc_3D, lab_trk_new_conc_3D, epochs, nVtxFeats, nTrackFeats, loss)
        stop = time.time()
        print("new concat model trained  in " + str(stop-start) + " seconds")

        new_conc_pred, new_conc_pred1D, new_conc_pred_roc = utl.double_input_prediction(
            new_conc_model, new_conc_x_test_vtx, new_conc_x_test_trk)

        ### from 3D to 2D ###
        # new_conc_y_test_vtx = np.array(new_conc_y_test_vtx, dtype=float)
        # new_conc_y_test_vtx.resize(test_size, 1, 1)
        # new_conc_x_test_2D_trk = utl.threeD_to2D(new_conc_x_test_vtx, test_size, 6)
        # new_conc_y_test_2D_trk = utl.threeD_to2D(new_conc_y_test_vtx, test_size, 1)

        print(" ...concatenate() model plotting ... ")

        plot.plot_roc(new_conc_x_test_vtx, new_conc_y_test_vtx,
                      new_conc_model, new_conc_pred_roc, new_conc_roc_str, plot_path)
        plot.plot_history(new_conc_hist, new_conc_hst_str, plot_path)
        plot.plot_probs(new_conc_y_test_vtx, new_conc_pred1D,
                        new_conc_prob_str, plot_path)
        #plot_model(new_conc_model, to_file=model_path +  '/new_conc_model_nodesper10_100Epochs.png', show_shapes=True)
        plot_model(new_conc_model, to_file=model_path +
                   '/' + mod_str, show_shapes=True)

        print(" ...concatenate() model...  DONE!!!!!!  ")

    return None


models = {
    "vertex": 1,
    "tracks": 0,
    "vertex_lstm": 0,
    "tracks_lstm": 0,
    "conc": 0,
    "add": 0,
    "conc_best": 1
}


def main():
    nSamples = 110000
    test_size = 20000
    epochs = 100
    nFeats = 10
    nVtxFeats = 9
    nTrackFeats = 5
    loss = binary_crossentropy
    print("loss is : ", loss)
    epochs_name = '100Epochs'
    mod_name = '_100Epochs'
    #path = '/home/atlas/sspinali/Qual_task/Outputs/binary_crossentropy/10Var/'
    path = sys.argv[1]

    if not os.path.exists(path):
        os.makedirs(path)

    execute(nSamples,
            nVtxFeats,
            nTrackFeats,
            epochs,
            loss,
            epochs_name,
            test_size,
            path,
            mod_name,
            models)

    return None


main()
