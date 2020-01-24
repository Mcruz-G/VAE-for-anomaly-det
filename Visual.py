import matplotlib.pyplot as plt
from matplotlib import rc


def viewLoss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

def viewFeatures(originalImg, encodedImg, predictedImg):

    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].plot(originalImg[i])
        axs[0, i].set_title('Input Signal')
        axs[1, i].plot(predictedImg[i], 'tab:green')
        axs[1, i].set_title('Reconstructed Signal')
        
    plt.show()

def viewAnomalies(probVec,probMat):
    plt.style.use('seaborn')
    width = 345
    
    fig_1 = plt.plot(probVec,'bs')
    plt.title('Minimum Reconstruction Probability per Colada')
    plt.xlabel('Colada')
    plt.ylabel(r'$RP_{i,t}$')
    plt.show()

    fig_2 = plt.plot(probMat[0][:],)
    plt.title('Reconstruction Probability evolution on regular Colada')
    plt.xlabel(r'$Time(t)$')
    plt.ylabel(r'$RP_{0,t}$')
    plt.show()

    fig_3 = plt.plot(probMat[5][:],)
    plt.title('Reconstruction Probability evolution on anomalous Colada')
    plt.xlabel(r'$Time(t)$')
    plt.ylabel(r'$RP_{5,t}$')
    plt.show()