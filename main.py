import numpy as np 
from Data_Processing import get_data, normalize
import numpy as np
from Loss import globalLoss
import matplotlib.pyplot as plt
from scipy.stats import norm
from Visual import viewLoss, viewFeatures, viewAnomalies
# import keras.losses
from keras.regularizers import l1, l2
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
# from keras.utils import plot_model
from keras import metrics
from keras.datasets import mnist
from anomaly import anomalyIndicator


######### Data process ###############


## Create a list of file names for each colada

nameList = ['MOD10076','MOD10312','MOD10825','MOD10832','MOD10899','MOD11234','MOD12544','MOD12701','MOD12800','MOD12868',
            'MOD14004','MOD14040','MOD14163','MOD14201','MOD14258','MOD14271','MOD14317','MOD14369','MOD14412','MOD14434',
            'MOD14452','MOD14468','MOD14487','MOD14551','MOD14567','MOD14672'
            ,'MOD14688','MOD14748','MOD14846','MOD14931','MOD14941','MOD15097']

nameList_test = ['MOD19890','MOD19895','MOD10156','MOD10765','MOD10784','MOD11054','MOD11246','MOD11563','MOD12628',
            'MOD12847','MOD13521','MOD14166','MOD14284','MOD14281','MOD14291','MOD14337','MOD14546','MOD14627','MOD14668','MOD14654',
            'MOD14666','MOD14681','MOD14701','MOD14713','MOD14722','MOD14718','MOD14734',
            'MOD14755','MOD14752','MOD14842','MOD14849','MOD14843','MOD14889','MOD14913','MOD15029']

## Create a zeros tensor to save each multivariate time-series
## like a bunch of images
dummy = get_data('MOD10076')

dataSet = np.zeros((len(namdiceseList),dummy.shape[0]))   #,dummy.shape[1]
dataSet_test = np.zeros((len(nameList_test),dummy.shape[0]))     #,dummy.shape[1]

## Extract the colada's data from local file
colada = 0                                          
for name in nameList:
    trainingExample = get_data(name)       
    dataSet[colada,:,] = trainingExample
    colada += 1

# Extract test data
colada = 0                                          
for name in nameList_test:
    trainingExample = get_data(name, test = True)       
    dataSet_test[colada,:,] = trainingExample
    colada += 1

## Proceed to normalize data
X_train = normalize(dataSet)
# reshape data to get
X_train = X_train.reshape(len(X_train), X_train.shape[1])

## Proceed to normalize data
X_test = normalize(dataSet_test)
# reshape data to get
X_test = X_test.reshape(len(X_test), X_test.shape[1])

###########################################

### Some Hyperparameters:
batchSize = 200  #Batch size for NN training
originalDim = dummy.shape[0]  #* dummy.shape[1] #Input's dimension 
latentDim = 1     #Latent layer dimension (features extracted layer)
intermediateDim = 256  #Hidden layer dimension
epochs = 300    #Training epochs
epsilonStd = 1.0  #Sigma for semi-stochastic node

####### Neural Network Architecture ##########

## Build encoder model

inputLayer = Input(shape = (originalDim,)) 
hiddenLayer_1 = Dense(units = 300, activation = 'relu',bias_regularizer = l1(0.001))(inputLayer)
hiddenLayer_2 = Dense(units = 256, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_1)
hiddenLayer_3 = Dense(units = 128, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_2)
hiddenLayer_4 = Dense(units = 64, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_3)
hiddenLayer_5 = Dense(units = 32, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_4)
hiddenLayer_6 = Dense(units = 16, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_5)
hiddenLayer_7 = Dense(units = 16, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_6)
hiddenLayer_8 = Dense(units = 16, activation = 'relu',bias_regularizer = l1(0.001))(hiddenLayer_7)

mu = Dense(latentDim, activation = 'linear')(hiddenLayer_8)
sigma = Dense(latentDim, activation = 'linear')(hiddenLayer_8)

# Here's the sampling function for the semi-stochastic node.

def sampling(args):
    mu, sigma = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], 1), mean=0.,
                              stddev=1.0)
    return mu + K.exp(sigma / 2) * epsilon

#Sampling Layer. 

z = Lambda(sampling, output_shape = (1,))([mu, sigma])

#Instantiate encoder model

encoder = Model(inputLayer,[mu,sigma,z], name = 'encoder')
encoder.summary()



### Build decoder model

latentInput = Input(shape = (latentDim,), name = 'z_sampling')

decoderHidden_1 = Dense(16, activation = 'relu',bias_regularizer = l1(0.001))(latentInput)
decoderHidden_2 = Dense(16, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_1)
decoderHidden_3 = Dense(16, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_2)
decoderHidden_4 = Dense(32, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_3)
decoderHidden_5 = Dense(64, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_4)
decoderHidden_6 = Dense(units = 128, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_5)
decoderHidden_7 = Dense(units = 256, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_6)
decoderHidden_8 = Dense(units = 300, activation = 'relu',bias_regularizer = l1(0.001))(decoderHidden_7)
decoderMean = Dense(originalDim, activation = 'sigmoid',bias_regularizer = l1(0.001))(decoderHidden_8)



#Instantiate decoder
decoder = Model(latentInput,decoderMean,name = 'decoder')

decoder.summary()
outputLayer = decoder(encoder(inputLayer)[2])
###############################################################

#Instantiate VAE model

vae = Model(inputLayer, outputLayer)

#Add the loss function
# keras.losses.globalLoss = globalLoss(originalDim, inputLayer, outputLayer, mu, sigma,1,1)
vaeLoss = globalLoss(originalDim, inputLayer, outputLayer, mu, sigma,1,1)
vae.add_loss(vaeLoss)
vae.compile(optimizer='rmsprop', metrics = ['acc'])
vae.summary()


#Fit the model
history = vae.fit(X_train, shuffle = True, epochs=epochs, batch_size=batchSize)

# Loss evolution
viewLoss(history)

encodedSeries = encoder.predict(X_test)
predictedSeries = vae.predict(X_test)

viewFeatures(X_test,encodedSeries, predictedSeries)


################ Tests ################################

# Create a test signal that should be 'Anomalous'

X_test[5][:] = 0.0
# X_test[5][:] = 0.01*np.random.rand(X_test.shape[1])
X_test[15][:] = 0.0
X_test[25][:] = 0.0
probVec, probMat = anomalyIndicator(encoder, decoder,X_test,nameList_test,alpha = 0.1,sampleNum =2)

# Visualize reconstruction probability
viewAnomalies(probVec, probMat)
