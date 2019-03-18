

def twoD_to_3D(features, labels, batch_size, timesteps, input_dim):
    import numpy as np
    
    print("----- before 3D reshaping -----")
    print("feat and lab shape :", features.shape, labels.shape)
    features_3D = np.array(features, dtype=float)
    labels_3D = np.array(labels, dtype=float)

    features_3D.resize(batch_size, timesteps, input_dim)
    labels_3D.resize(batch_size, timesteps, 1)
    
    print("----- after 3D reshaping -----")
    print("feat and lab shape :", features_3D.shape, labels_3D.shape)
    
    return features_3D, labels_3D


def threeD_to2D(vect, nSamples, nDim):
    import numpy as np

    print(" ## BEFORE ## , vect shape and dimensions :", vect.shape, vect.ndim)
    vect = np.array(vect, dtype=int)
    #vect = np.array(vect)
    vect.resize(nSamples, nDim)
    print(" ## AFTER ## , vect shape and dimensions :", vect.shape, vect.ndim)
    
    return vect
