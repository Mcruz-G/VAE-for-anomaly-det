
from keras import backend as K
from keras import metrics
from keras.losses import mse, binary_crossentropy

def globalLoss(initialDim,inputs,outputs,mu,sigma, alpha, beta):
    ########### Loss function #######################################

    ## There're two loss functions which constitute the global Loss function,
    ## In esence, this neural network is trying to minimize 
    ## the reconstruction loss and the KL-Divergence.

    #Reconstruction loss
    recLoss = initialDim * mse(inputs, outputs)

    #KL-divergence loss function
    klLoss = -0.5 * K.sum( 1 + sigma - K.square(mu)- K.exp(sigma), axis = -1)

    #Global Loss Function
    vaeLoss = K.mean(alpha*recLoss + beta*klLoss)

    return vaeLoss