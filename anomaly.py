import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
from Visual import viewAnomalies
import math



def normpdf(x, mean, sd):
    var = sd**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-1/2*(x-mean)**2)
    return num/denom

def anomalyIndicator(encoder, decoder, X_test, nameTest, alpha, sampleNum = 100):

    recProbMat = np.zeros((X_test.shape[0], X_test.shape[1]))
    recProbVec = np.zeros((X_test.shape[0]))
    
    for i in range(X_test.shape[0]):
          

        #Estimate new data's mean in latent space
        mu_Z= encoder.predict(X_test[i:i+1])[0] 

        #Estimate new data's variance in latent space
        sigma_Z = encoder.predict(X_test[i:i+1])[1] 

        # Draw sampleNum samples:
        Z = np.random.normal(mu_Z[0],(sigma_Z[0])**2,sampleNum)

        for t in range(X_test.shape[1]):
            recProb = 0 probMa
            for l in range(sampleNum):
                mu_X = decoder.predict(np.array([Z[l]]))
                prob = norm.pdf(X_test[i:i+1][0][t],mu_X[0][t], 1)
                recProb += prob  

            
            recProbMat[i][t] = (2*math.pi)**0.5*recProb/sampleNum
        
        recProbVec[i] = min(recProbMat[i][:])

    return recProbVec, recProbMat

    