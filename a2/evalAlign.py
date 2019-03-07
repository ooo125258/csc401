#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

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
        alignment_model = align_ibm1(data_dir, num_sent, max_iter)
    return alignment_model
    #pass


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
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
    return [BLEU_score(eng_decoded[i], [eng[i], google_refs[i]], n) for i in range(len(eng_decoded))]
    #pass
   

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
    sTask5e = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e"
    sTask5f = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"
    sTask5googlee = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"
    sData_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    sFn_LM = "fn_LM"
    sFn_AM = "fn_AM"
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
        print("Warning: following file failed to open: " + fTask5googlee)
        lGoogle_eng = []

    #Read Lm
    dLM = _getLM(sData_dir, 'e', sFn_LM, not args.force_refresh)
    
    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Am is the number of iterations. dAM is the dict of AM.
        #As 25*4*3, the iteration number is 3
        dAM = align_ibm1(sData_dir, AM, 3, sFn_AM)
        #25 sentences:
        for i in range(25):
            lSent_prep_fre = preprocess(lHansard_fre[i], 'f')
            # Eval using 3 N-gram models #
            all_evals = []
            for n in range(1, 4):
                lDecoded_fre = decode(lSent_prep_fre, dLM, dAM)
                sHanzard_prep_ref_eng = preprocess(lHansard_eng[i], 'e')
                sGoogle_prep_ref_eng = preprocess(lGoogle_eng[i], 'e')
                                
                f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
                evals = _get_BLEU_scores(lDecoded_fre, [sHanzard_prep_ref_eng, sGoogle_prep_ref_eng], n)
                for v in evals:
                    f.write(f"\t{v:1.4f}")
                all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    parser.add_argument("-f", "--force_refresh", action="store_true", help="Use saved cached value to run to accelerate")
    args = parser.parse_args()
    
    main(args)
