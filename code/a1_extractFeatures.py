import numpy as np
import sys
import argparse
import os
import json
from collections import defaultdict
from numpy import loadtxt

def init():

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    print('TODO')
    # TODO: your code here
    PRP_1st = loadtxt("/u/cs401/Wordlists/First-person", comments="#", delimiter="\n", unpack=False)
    PRP_1st = loadtxt("/u/cs401/Wordlists/Second-person", comments="#", delimiter="\n", unpack=False)
    PRP_1st = loadtxt("/u/cs401/Wordlists/Third-person", comments="#", delimiter="\n", unpack=False)
    #Now, start to check the words
    sentences = comment.split("\n")

    numbers = {}
    for sentence in sentences:
        #to avoid some silly empty string.
        if sentence == "" or sentence is None:
            continue

        words = sentence.split()
        # 1, first person pronoun
        for i in range(len(words)):


def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

