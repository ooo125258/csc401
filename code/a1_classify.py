from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import csv
import random
def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / np.sum(C)
    #print ('TODO')

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k,k] / np.sum(C[k]) for k in range(C.shape[0])]
        
    #print ('TODO')

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[k,k] / np.sum(C[:,k]) for k in range(C.shape[0])]
    #print ('TODO')
    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    feats = np.load(filename)#To easier to debug, random_state=0. change later
    X_train, X_test, y_train, y_test = train_test_split(feats[:,:173], feats[:,173], test_size = 0.2, random_state=0)

    compare_values = np.zeros((5,26))
    compare_values[:0] = [1,2,3,4,5]

    #1 linearSVC
    lsvc_clf = LinearSVC(random_state=0, loss="hinge")
    lsvc_clf.fit(X_train, y_train)
    y_pred_1 = lsvc_clf.predict(X_test)
    C1 = confusion_matrix(y_test, y_pred_1)
    compare_values[0][1] = accuracy(C1)
    compare_values[0][2:6] = recall(C1)
    compare_values[0][6:10] = precision(C1))
    compare_values[0][10:] = C1.reshape(16,)

    #2 SVC
    svc_clf = SVC(gamma=2)
    svc_clf.fit(X_train, y_train)
    y_pred_2 = svc_clf.predict(X_test)
    C2 = confusion_matrix(y_test, y_pred_2)
    compare_values[1][1] = accuracy(C2)
    compare_values[1][2:6] = recall(C2)
    compare_values[1][6:10] = precision(C2))
    compare_values[1][10:] = C2.reshape(16,)

    #3 RFC
    rfc_clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    rfc_clf.fit(X_train, y_train)
    y_pred_3 = rfc_clf.predict(X_test)
    C3 = confusion_matrix(y_test, y_pred_3)
    compare_values[2][1] = accuracy(C3)
    compare_values[2][2:6] = recall(C3)
    compare_values[2][6:10] = precision(C3))
    compare_values[2][10:] = C3.reshape(16,)

    #4 MLP
    mlp_clf = MLPClassifier(alpha=0.05)
    mlp_clf.fit(X_train, y_train)
    y_pred_4 = mlp_clf.predict(X_test)
    C4 = confusion_matrix(y_test, y_pred_4)
    compare_values[3][1] = accuracy(C4)
    compare_values[3][2:6] = recall(C4)
    compare_values[3][6:10] = precision(C4))
    compare_values[3][10:] = C4.reshape(16,)

    #5 AdaBoost
    ab_clf = AdaBoostClassifier()
    ab_clf.fit(X_train, y_train)
    y_pred_5 = ab_clf.predict(X_test)
    C5 = confusion_matrix(y_test, y_pred_5)
    compare_values[4][1] = accuracy(C5)
    compare_values[4][2:6] = recall(C5)
    compare_values[4][6:10] = precision(C5))
    compare_values[4][10:] = C5.reshape(16,)

    #Go, iBest!
    iBest = np.argmax(compare_values[:,1]) + 1

    #TODO: for debug use:
    print("3.1 compare_values")
    print(compare_values)

    np.savetxt("a1_3.1.csv", compare_values, delimiter=",")

    #Backup values for later use
    np.savez('a31retvalues.npz', name1=X_train, name2=X_test, name3=y_train, name4=y_test, name5=iBest)
    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    train_nums = [1000, 5000, 10000, 15000, 20000]
    compare_values = []
    X_1k = None
    y_1k = None
    for train_num in train_nums:
        selected_value_index = random.sample(range(len(X_train)), train_num)
        reduced_X_train = X_train[selected_value_index]
        reduced_Y_train = Y_train[selected_value_index]

        if train_num == 1000:
            X_1k = reduced_X_train
            y_1k = reduced_Y_train
        clf = None
        if i == 1:
            clf = LinearSVC(random_state=0, loss="hinge")
        elif i == 2:
            clf = SVC(gamma=2)
        elif i == 3:
            clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        elif i == 4:
            clf = MLPClassifier(alpha=0.05)
        elif i == 5:
            clf = AdaBoostClassifier()
        else:
            print("Critical ERROR! classifier is not identified! You may stop!")
            print("It will continue as LinearSVC, but it might not be the highest!!!")
            clf = LinearSVC(random_state=0, loss="hinge")

        clf.fit(reduced_X_train, reduced_y_train)
        y_pred = clf.predict(X_test)
        C = confusion_matrix(y_test, y_pred)
        compare_values.append(accuracy(C))

    print("a1 3.2")
    print(compare_values)
    with open("a1_3.2.csv", "wb") as csvfile:
        a132writer = csv.writer(csvfile, delimiter=',')
        a132writer.writerow(compare_values)
    

    #Backup values for later use
    np.savez('a31retvalues.npz', name1=X_train, name2=X_test, name3=y_train, name4=y_test, name5=iBest)
    
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nothing. Part3')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    print("Classifier done.")