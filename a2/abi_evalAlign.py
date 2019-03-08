from abi_BLEU_score import *
from decode import *
from preprocess import *
from align_ibm1 import *
from lm_train import *
import pickle

def evalAlign(hansard_english, hansard_french, google_english, LM_PATH, train_dir, fn_AM, report_path):
    """
    Evaluates the alignment model created by IBM-1 algorithm by comparing french to english translations from AM
    with translations from the hansard_english and google_english files using BLEU_score
    hansard_english: /Hansard/Testing/Task5.e
    hansard_french: /Hansard/Testing/Task5.f
    google_english: /Hansard/Testing/Task5.google.e
    LM_PATH: English language_model path
    train_dir: directory for training set
    fn_AM: path to save AM
    report_path: path to save Task5.txt report
    """
    # which file to write the report
    report_file = open(report_path, 'w')

    # read in all the sentences
    hansard_english = open(hansard_english).read().split('\n')
    hansard_french = open(hansard_french).read().split('\n')
    google_english = open(google_english).read().split('\n')

    # create in the language model
    LME = lm_train(train_dir, "e", LM_PATH)

    # decode and calculate blue score for AM models trained on different num_sentences
    # and BLEU_scores calculated on different n-grams
    for num_sentences in [1000, 10000, 15000, 30000]:
        AM = align_ibm1(train_dir, num_sentences, 5, fn_AM)
        for f in range(25):
            fproc = preprocess(hansard_french[f], 'f')
            for n in range(1,4):
                my_english = decode(fproc, LME, AM)

                hans_ref = preprocess(hansard_english[f], 'e')
                ggle_ref = preprocess(google_english[f], 'e')
                references = [hans_ref, ggle_ref]
                print('MY-CANDIDATE: {}\nHANS-REF: {}\nGOOGLE-REF: {}'.format(my_english, hans_ref, ggle_ref))

                score = BLEU_score(my_english, references, n)
                report_string = 'french-sentence: {}, my-english-sentence: {}, num-sentences-for-AM: {}, n-grams: {}, BLEU-score: {}'.format(fproc, my_english, num_sentences, n, score)
                report_file.write(report_string + '\n')

    report_file.close()

if __name__ == "__main__":
    hansard_english = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e'
    hansard_french = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f'
    google_english = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e'
    saved_files = '//h/u15/c4/00/sunchuan/csc401/a2'
    LM_PATH = '{}/LM_e'.format(saved_files)
    train_dir = '/u/cs401/A2_SMT/data/Hansard/Training/'
    fn_AM = '{}/AM_test'.format(saved_files)
    report_path = '{}/Task5_abitest.txt'.format(saved_files)
    evalAlign(hansard_english, hansard_french, google_english, LM_PATH, train_dir, fn_AM, report_path)
