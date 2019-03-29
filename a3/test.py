from a3_gmm import *
from scipy import stats
from tqdm import tqdm
import pylab
import os
import fnmatch


def execute(M, maxIter, maxSpeaker, f):
    trainThetas = []
    testMFCCs = []
    
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    epsilon = 0.0
    
    trainThetas = []
    testMFCCs = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in tqdm(dirs):
            print("Speaker: {}, Iteration: {} M:{}")
            print(speaker, file=f)
            
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)
            
            if s < maxSpeaker:
                testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                testMFCCs.append(testMFCC)
                
                # if s < maxSpeaker:
                X = np.empty((0, d))
                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X = np.append(X, myMFCC, axis=0)
                
                trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            else:
                trainThetas.append(theta(speaker))
            s += 1
    
    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print('Accuracy: ', accuracy, file=f)
    print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
    print('\n', file=f)
    
    # write to file
    print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))



if __name__ == "__main__":

    random.seed(0)
    #stats.probplot(measurements, dist="norm", plot=pylab)
    pylab.show()
    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    epsilon = 0.0
    maxIter = 20
    M = 8
    # maxSpeaker = 32
    f = open('gmm_S.txt', 'w')

    M_list = [20, 15, 10, 8, 5, 2, 1][::-1]
    maxIter_list = [50, 30, 20, 15, 10, 5, 2, 1][::-1]
    maxSpeaker_list = [32, 20, 15, 10, 5, 2, 1][::-1]

    for M in tqdm(M_list):
        execute(M, 20, 100)
    for maxIter in tqdm(maxIter_list):
        execute(8, maxIter, 100)
    for maxSpeaker in tqdm(maxSpeaker_list):
        execute(8, 20, maxSpeaker)
    
    f.close()

