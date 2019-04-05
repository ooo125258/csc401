from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import logsumexp
import os, fnmatch
import random
from tqdm import tqdm
import copy
dataDir = '/u/cs401/A3/data/'
np.random.seed(seed=131)

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handoutd

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    #Format pt.2. Term2 might be preComputed
    #TODO: need to debug when the train is permitted. At least the np dots.
    M, d = myTheta.Sigma.shape
    mask = myTheta.Sigma[m] != 0
    inv_sigmaSqr = np.array(myTheta.Sigma[m], copy=True)
    inv_sigmaSqr[mask] = np.reciprocal(myTheta.Sigma[m][mask])
    square_x = np.square(x).T    
    term1_1 = 0.5 * np.dot(inv_sigmaSqr, square_x)
    term1_2 = np.sum(myTheta.mu[m] * inv_sigmaSqr * x)
    term1_beforesum = term1_1 - term1_2
    term1 = np.sum(term1_beforesum)# TODO: axis of term1?

    term2 = 0
    # checker:# Assume, precomputedform is a m len list.
    if len(preComputedForM) != myTheta.Sigma.shape[0] \
            or len(np.array(preComputedForM).flatten()) != myTheta.Sigma.shape[0]:
        
        term2_1_inner = 0.5 * np.square(myTheta.mu[m]) * inv_sigmaSqr
        term2_1 = np.sum(term2_1_inner)
        term2_2 = d / 2 * np.log(2 * np.pi)
        term2_3_inner = 0.5 * np.log(myTheta.Sigma[m], where=(myTheta.Sigma[m] != 0.))
        term2_3 = np.sum(term2_3_inner)# TODO: check when sigma is not 1, if it still performs correctly.
        term2 = term2_1 + term2_2 + term2_3
    else:
        term2 = preComputedForM[m]
    
    rst = - term1 - term2
    return rst

def log_b_m_x_given(x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handoutd

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    #In vector form, precompute part would be slower. As more steps need to be made.
    #TODO: need to debug when the train is permitted. At least the np dots.
    M, d = myTheta.Sigma.shape
    T = x.shape[0]
    
    mask = myTheta.Sigma != 0
    inv_sigmaSqr = np.array(myTheta.Sigma, copy=True)
    inv_sigmaSqr[mask] = np.reciprocal(myTheta.Sigma[mask])
    term1_wosum = np.zeros((M,T))
    for m in range(M):
        term1_m = 0.5 * np.square(x - myTheta.mu[m]) #T * d - d = T*d
        term1_wosum[m] = np.dot(term1_m, inv_sigmaSqr[m]) #T*d, d = T
    term1 = term1_wosum #M * T, M
    term2 = d / 2 * np.log(2 * np.pi)
    term3_inner = 0.5 * np.log(myTheta.Sigma, where=(myTheta.Sigma != 0.))
    term3 = np.sum(term3_inner, axis=1)  # TODO: check when sigma is not 1, if it still performs correctly.
    rst = -((term1.T + term3.T).T + term2)
    
    return rst

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    #I cannot use log a + log b - logc form because some of them might be negative!
    M, d = myTheta.Sigma.shape
    numo = np.dot(myTheta.omega[m], np.exp(log_b_m_x(m, x, myTheta)))
    deno = 0
    for k in range(M):
        deno += np.dot(myTheta.omega[k], np.exp(log_b_m_x(k, x, myTheta)))
    if deno == 0. or numo == 0:
        return 0.
    rst = np.log(numo / deno)
    return rst

def log_p_m_x_given(  m, log_Bs, t, myTheta, preComputedForM = []):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        Given Bs
    '''
    M, d = myTheta.Sigma.shape
    logOmega = np.log(myTheta.omega, where=(myTheta.omega != 0.))
    log_WBs = logOmega[m] + log_Bs[m,t]
    denos = logOmega + log_Bs[:, t].reshape(-1, 1)
    rst = log_WBs - logsumexp(denos)
    
    return rst
    
    
def logLik( log_Bs, myTheta ):
    # Todo: should we return a sum? for whole M*T matrix?
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    log_Ws = np.log(myTheta.omega, where=(myTheta.omega != 0.))
    rst = logsumexp(log_Bs + log_Ws, axis=0) # TODO:logsumexp
    return np.sum(rst)

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
#Input: MFCC data X, number of components M , threshold , and maxIter
    #Ls = [] # TODO: DELETE
    #Initialize θ ;
    myTheta = theta( speaker, M, X.shape[1] )
    #i := 0 ;
    i = 0
    T = X.shape[0]
    M, d = myTheta.Sigma.shape
    myTheta.Sigma[:,:] = 1
    #indices = np.random.choice(X.shape[0], M, replace=False)
    indices = np.array(random.sample(range(T), M))
    myTheta.mu = X[indices]
    myTheta.omega[:, 0] = 1. / M
    
    logLik_array = np.zeros((M, T))
    #prev L := −∞ ; improvement = ∞;
    prev_L = np.NINF
    improvement = np.Inf

    log_Bs = np.zeros((M, T))
    log_Ps = np.zeros((M, T))
    #while i =< maxIter and improvement >=  do
    while i < maxIter and improvement > epsilon:
        precomputed = preComputedForEachM(myTheta)
        #log_b_m_x(0, X[0], myTheta, precomputed) #TODO: wrong
        
        log_Bs = log_b_m_x_given(X, myTheta, precomputed)
        
        log_WBs = log_Bs + np.log(myTheta.omega, where=(myTheta.omega != 0.))
        log_Ps = log_WBs - logsumexp(log_WBs, axis=0)
        #for m in range(M):
        #    for t in tqdm(range(T)):
        #        log_Ps[m, t] = log_p_m_x(m, X[t], myTheta)
        Ps = np.exp(log_Ps)
        
    #    ComputeIntermediateResults ;
    #    L := ComputeLikelihood (X, θ) ;
        if np.sum(myTheta.Sigma == 0.):
            print("Training")
            raise IndexError
        L = logLik(log_Bs, myTheta)
    #    θ := UpdateParameters (θ, X, L) ;
        myTheta = UpdateParameters(myTheta, X, Ps, L)
    #    improvement := L − prev L ;
        improvement = L - prev_L
    #    prev L := L ;
        prev_L = L
    #    i := i + 1 ;
        #Ls.append(L)#TODO: DELETE
        i += 1
    #end
    return myTheta#, Ls #TODO: DELETE

def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
        mfcc is sth like x, models is sth like list of mytheta
        #NOTICE: This is not the correct format, just for me to debug!!!
    '''
    bestModel = -1
    M, d = models[0].Sigma.shape
    T = mfcc.shape[0]

    Ls = np.zeros((len(models)))
    for i in range(len(models)):
        precomputed = preComputedForEachM(models[i])
        log_Bs = np.zeros((M, T))
        #for m in range(M):
            #for t in range(T):
                #log_Bs[m, t] = log_b_m_x(m, mfcc[t], models[i], precomputed)
        log_Bs = log_b_m_x_given(mfcc, models[i])
        if np.sum(models[i].Sigma == 0.):
            print("Testing")
        #    raise IndexError
        Ls[i] = logLik(log_Bs, models[i])
    if k > 0:
        Desc_order = np.argsort(Ls)[::-1]
        best_k = Desc_order[:k]
        bestModel = Desc_order[0]
        '''
        print("The correct model should be {}.".format(models[correctID].name))
        print("The best model might be {}.".format(models[bestModel].name))
        for i in range(len(best_k)):
            print("Model: {} Likelihood: {}".format(models[best_k[i]].name, Ls[best_k[i]]))
        '''
        print("\n" + models[correctID].name)
        for i in range(len(best_k)):
            print(models[best_k[i]].name + " " + str(Ls[best_k[i]]))
        #print("bestModel: {} correctID: {}".format(bestModel, correctID))
    return 1 if (bestModel == correctID) else 0#, Ls #TODO: DELETE


def preComputedForEachM(myTheta):
    # For term2 in log_Bs
    
    # preComputedForM
    M, d = myTheta.Sigma.shape
    mask = myTheta.Sigma != 0
    inv_sigmaSqr = np.array(myTheta.Sigma, copy=True)
    inv_sigmaSqr[mask] = np.reciprocal(myTheta.Sigma[mask])
    
    term2_1_inner = 0.5 * np.square(myTheta.mu) * inv_sigmaSqr
    term2_1 = np.sum(term2_1_inner, axis=1)
    term2_2 = d / 2 * np.log(2 * np.pi)
    
    term2_3_inner = 0.5 * np.log(myTheta.Sigma, where=(myTheta.Sigma != 0.))
    term2_3 = np.sum(term2_3_inner, axis = 1)# TODO: check when sigma is not 1, if it still performs correctly.
    term2s = term2_1 + term2_2 + term2_3
    return term2s


def UpdateParameters(myTheta, X, Ps, L):
    newTheta = copy.deepcopy(myTheta)
    T = X.shape[0]
    #Ps = np.exp(log_Ps)
    newTheta.omega = np.average(Ps, axis=1).reshape(-1,1)
    #Weighted mean, weight xt, mean p
    
    newTheta.mu = np.divide(np.dot(Ps, X), np.sum(Ps, axis=1).reshape(-1,1))# Todo: Wait, ps might be zero???

    square_x_term = np.divide(np.dot(Ps, np.square(X)), np.sum(Ps, axis=1).reshape(-1,1))
    newTheta.Sigma = square_x_term - np.square(newTheta.mu)
    return newTheta 

    
if __name__ == "__main__":
    load = False
    trainThetas = []
    testMFCCs = []
    #print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 1
    epsilon = 0.0
    maxIter = 20
    random.seed(0)
    if not load:
        # train a model for each speaker, and reserve data for testing
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in tqdm(dirs):
    
                files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
                random.shuffle( files )
                testMFCC = np.load( os.path.join( dataDir, speaker, files.pop(0) ) )
                testMFCCs.append( testMFCC )
    
                X = np.empty((0,d))
                for file in files:
                    myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                    X = np.append( X, myMFCC, axis=0)
    
                trainThetas.append( train(speaker, X, M, epsilon, maxIter) )
    
        # evaluate
        numCorrect = 0;
        import pickle
        with open('outfile', 'wb') as fp:
            pickle.dump(trainThetas, fp)
        for i in tqdm(range(0,len(testMFCCs))):
            numCorrect += test( testMFCCs[i], i, trainThetas, k )
        accuracy = 1.0*numCorrect/len(testMFCCs)
        print(accuracy)
    else:
        import pickle
        with open('outfile', 'rb') as fp:
            trainThetas = pickle.load(fp)
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in tqdm(dirs):
                print(speaker)
        
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)
        
                testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                testMFCCs.append(testMFCC)
        numCorrect = 0;
        for i in tqdm(range(0,len(testMFCCs))):
            numCorrect += test( testMFCCs[i], i, trainThetas, k )
        accuracy = 1.0*numCorrect/len(testMFCCs)
        print(accuracy)

