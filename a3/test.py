from a3_gmm import *
from scipy import stats
from tqdm import tqdm
import pylab
import os
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
def execute(M, maxIter, maxSpeaker, f):
    trainThetas = []
    testMFCCs = []
    
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    epsilon = 0.0
    
    trainThetas = []
    testMFCCs = []
    s = 0
    train_Ls = np.zeros((min(maxSpeaker, 32), maxIter))
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in tqdm(dirs):
            #print("Speaker: {}, Iteration: {} M:{}")
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
                rst = train(speaker, X, M, epsilon, maxIter)
                train_Ls[s][:len(rst[1])] = rst[1][:]
                trainThetas.append(rst[0])
            else:
                #trainThetas.append(theta(speaker))
                pass
            s += 1
    
    # evaluate
    numCorrect = 0
    logL = np.zeros((min(maxSpeaker, 32), min(maxSpeaker, 32)))
    for i in range(0, len(testMFCCs)):
        rst = test(testMFCCs[i], i, trainThetas, k)
        numCorrect += rst[0]
        logL[i][:] = rst[1][:]
    
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print('Accuracy: ', accuracy, file=f)
    print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
    print('\n', file=f)
    
    # write to file
    print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
    return logL,train_Ls #logL is model * model size L. When Testing model i, the llk for model j


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
    temp, train_Ls = execute(1, 100, 100, f)
    temp, train_Ls2 = execute(8, 100, 100, f)
    
    #for maxSpeaker in tqdm(maxSpeaker_list):
    #    print("Testing: M:{}, maxiter: {}, maxSpeaker: {}".format(8, 20, maxSpeaker))
    #    execute(8, 20, maxSpeaker, f)
    llk_M = np.zeros((len(M_list), 32, 32))
    for i in tqdm(range(len(M_list))):
        print("Testing: M:{}, maxiter: {}, maxSpeaker: {}".format(M, 20, 100))
        llk_M[i], temp= execute(M_list[i], 1, 100, f)
    print("llk_M get.")
    llk_I = np.zeros((len(maxIter_list), 32, 32))
    for i in tqdm(range(len(maxIter_list))):
        print("Testing: M:{}, maxiter: {}, maxSpeaker: {}".format(8, maxIter, 100))
        llk_I[i], temp= execute(1, maxIter_list[i], 100, f)
    print("llk_I get.")
    temp, train_Ls = execute(1, 100, 100, f)
    llk_M = -llk_M
    llk_I = -llk_I
    for i in range(llk_M.shape[1]):
        print(i)
        fig = plt.figure()
        plt.title("When testing model {}, with M".format(i))
        control_list = [0] * llk_M.shape[1]
        for j in range(llk_M.shape[2]):
            control_list[j] = plt.plot(llk_M[:,i,j], label=j)
        plt.legend(range(llk_M.shape[2]), ncol=2)
        plt.savefig("llkMformodel{}.png".format(i))
        plt.close()
        fig = plt.figure()
        plt.title("When testing model {}, with Max iteration".format(i))
        control_list = [0] * llk_I.shape[1]
        for j in range(llk_I.shape[2]):
            control_list[j] = plt.plot(llk_I[:,i,j], label=j)
        plt.legend(range(llk_I.shape[2]), ncol=2)
        plt.savefig("llkIformodel{}.png".format(i))
        plt.close()
    fig = plt.figure()
    plt.title("training llk")
    control_list = [0] * len(train_Ls)
    print("last")
    train_Ls = -train_Ls
    for i in range(train_Ls.shape[0]):
        control_list[i] = plt.plot(train_Ls[i], label=i)
        plt.legend(range(train_Ls.shape[1]), ncol=2)
        #overfit?
    plt.savefig("trainingplot.png")
    plt.close()
    f.close()
    
    
