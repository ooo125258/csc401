from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    training_data = read_hansard(train_dir, num_sentences)
    
    # Initialize AM uniformly
    # initialize P(f | e)
    AM = initialize(training_data["eng"], training_data["fre"])
    
    # Iterate between E and M steps

    

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	Return:
	    training data: {'eng':[], 'fre':[]}
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    # TODO
    counter = 0
    training_set = {'eng' : [], 'fre' : []}
    for root, dirs, files in os.walk(train_dir, topdown=False):
        for file in files:
            if not (len(file) > 2 and file[-1] == 'e' and file[-2] == '.'):#.e
                continue
                
            e_fullName = os.path.join(train_dir, file)
            f_fullName = e_fullName[:-1] + 'f'
            if not os.path.exists(f_fullName):
                #To remove eng without fre
                continue
            e_file = open(e_fullName)
            f_file = open(f_fullName)
            
            e_readLine = e_file.readline()
            f_readLine = f_file.readline()
            
            while e_readLine:  # "" is false directly
                if not f_readLine:
                    continue
                training_set['eng'].append(preprocess(e_readLine, 'e'))
                training_set['fre'].append(preprocess(f_readLine, 'f'))
                counter += 1
                
                if counter >= num_sentences:
                    # The time is now
                    e_file.close()
                    f_file.close()
                    return training_set
                e_readLine = e_file.readline()
                f_readLine = f_file.readline()
            e_file.close()
            f_file.close()
    return training_set
                
                    
        

def initialize(eng, fre):
    """
    list: eng
    list: fre
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	# TODO
	
	
    
def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO