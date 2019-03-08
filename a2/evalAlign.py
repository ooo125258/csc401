#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle
import math
import numpy as np

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    language_model = None
    if use_cached:
        with open(fn_LM + '.pickle', 'rb') as handle:
            language_model = pickle.load(handle)
    else:
        language_model = lm_train(data_dir, language, fn_LM)
    return language_model
    #pass

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    alignment_model = None
    if use_cached:
        with open(fn_AM + '.pickle', 'rb') as handle:
            alignment_model = pickle.load(handle)
    else:
        alignment_model = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
    return alignment_model
    #pass


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Yes, I calculate TRUE blue score here!
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    #Warning! you need to fully debug what happened!
    lflRet = []
    for i in range(len(eng_decoded)):
        if i == 20:
            print (1)
        #BLEU_score will only return the p_n, when false
        lPs = np.zeros(n)
        iCandidate_tokens_len = len(eng_decoded[i].split())
        lsReferences = [eng[i], google_refs[i]]
        brevity_val = brevity(iCandidate_tokens_len, lsReferences)
        for j in range(n):
            if j == 0:
                print(1)
            lPs[j] = BLEU_score(eng_decoded[i], lsReferences, j + 1)
            #Warning, we assume it starts at STENSTART and end in STENEND.
        flScore = brevity_val * math.pow(np.prod(lPs), 1.0 / n)
        lflRet.append(flScore)
    return lflRet


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    AMs = [1000, 10000, 15000, 30000]
    #Read file:
    print("Start! Linking files:")
    sTask5e = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e"
    sTask5f = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"
    sTask5googlee = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"
    sData_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    sFn_LM = args.fn_LM
    sFn_AM = args.fn_AM
    try:
        lHansard_eng = open(sTask5e).read().split('\n')
    except IOError:
        print("Warning: following file failed to open: " + sTask5e)
        lHansard_eng = []
    try:
        lHansard_fre = open(sTask5f).read().split('\n')
    except IOError:
        print("Warning: following file failed to open: " + sTask5f)
        lHansard_fre = []
    try:
        lGoogle_eng = open(sTask5googlee).read().split('\n')
    except IOError:
        print("Warning: following file failed to open: " + sTask5googlee)
        lGoogle_eng = []

    #Read Lm
    print("GetLM:")
    dLM = _getLM(sData_dir, 'e', sFn_LM, not args.force_refresh)
    print("GetLM completed!")
    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    f = open(args.test5, 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")
    result = np.zeros((12,25)) #i * 3 + n, 25
    for i, AM in enumerate(AMs):
        print("\n### Evaluating AM model: {} ### \n".format(AMs[i]), file=f)
        # Decode using AM #
        # Am is the number of iterations. dAM is the dict of AM.
        #As 25*4*3, the iteration number is 10 #TODO: test this value
        print("GetAM for data size " + str(AM))
        dAM = _getAM(sData_dir, AM, 10, sFn_AM+str(AM), not args.force_refresh)
        #dAM = align_ibm1(sData_dir, AM, 10, sFn_AM)
        print("GetAM for data size " + str(AM) + " completed!")
        #25 sentences: 25 * each sentence, pre-handle first, to avoid extra preprocesses
        lsSent_prep_fre = []
        llDecoded_fre = []
        lsHanzard_prep_ref_eng = []
        lsGoogle_prep_ref_eng = []
        for sent_idx in range(25):
            sSent_prep_fre = preprocess(lHansard_fre[sent_idx], 'f')
            lsSent_prep_fre.append(sSent_prep_fre)
            # Eval using 3 N-gram models #
            all_evals = []
            llDecoded_fre.append(decode.decode(sSent_prep_fre, dLM, dAM))
            lsHanzard_prep_ref_eng.append(preprocess(lHansard_eng[sent_idx], 'e'))
            lsGoogle_prep_ref_eng.append(preprocess(lGoogle_eng[sent_idx], 'e'))
        for sent_idx in range(25):
            print("Candidate: {}\nHanzard: {}\nGoogle: {}\n".format(llDecoded_fre[sent_idx], lsHanzard_prep_ref_eng[sent_idx], lsGoogle_prep_ref_eng[sent_idx]))
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            #for sent_idx in range(25):
            evals = _get_BLEU_scores(llDecoded_fre, lsHanzard_prep_ref_eng, lsGoogle_prep_ref_eng, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)
        print("data size " + str(AM) + " Finished!")
        f.write("\n\n")
    #print(result, file=f)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    
    pass

def break_references(references):
    '''
    break each reference to piece, remove the STEN
    '''
def brevity(iCandidate_tokens_len, references):
    '''
    
    :param iCandidate_tokens_len: the length of candidate tokens
    :param references: [hanzard, google]
    :return: float, brevity
    '''
    brevity_val = 0
    #Warning! We assume it's embedded by STEN!
    sReal_ref = [ref.split() for ref in references]
    #Find the nearest length
    liLen_ref = [len(sReal_ref[i]) for i in range(len(sReal_ref))]
    #https:stackoverflow.com/questions/9706041
    iR_pos = min(range(len(liLen_ref)), key=lambda i : abs(liLen_ref[i] - iCandidate_tokens_len))
    if liLen_ref[iR_pos] <= iCandidate_tokens_len:#ri<ci
        brevity_val = 1
    else: #exp(1-ri/ci)
        brevity_val = math.exp(1 - liLen_ref[iR_pos] / iCandidate_tokens_len)
    return brevity_val

if __name__ == "__main__":
    print("Sample Test:")
    candidate1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
    candidate2 = "It is to insure the troops forever hearing the activity guidebook that party direct"
    references = ["It is a guide to action that ensures that the military will forever heed Party commands", "It is the guiding principle which guarantees the military forces always being under command of the Party", "It is the practical guide for the army always to heed the directions of the party"]
    n = 1
    print(BLEU_score(candidate1, references, n))
    n = 2
    print(BLEU_score(candidate1, references, n))
    print(BLEU_score(candidate2, references, n))
    candidate3 = "I fear David"
    ref = ["I am afraid Dave", "I am scared Dave", "I have fear David"]
    lPs = np.zeros(n)
    iCandidate_tokens_len = len(candidate3.split())
    lsReferences = ref
    brevity_val = brevity(iCandidate_tokens_len, lsReferences)
    print(brevity_val)
    for j in range(n):
        lPs[j] = BLEU_score(candidate3, lsReferences, j + 1)
        #Warning, we assume it starts at STENSTART and end in STENEND.
    print(lPs)
    flScore = brevity_val * math.pow(np.prod(lPs), 1.0 / n)
    print(flScore)
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    parser.add_argument("-r", "--force_refresh", action="store_true", help="Use saved cached value to run to accelerate")
    parser.add_argument("fn_LM", help="fn_LM")
    parser.add_argument("fn_AM", help="fn_AM")
    parser.add_argument("test5", help="test5")
    args = parser.parse_args()
    
    main(args)
    
