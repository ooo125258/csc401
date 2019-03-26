import os
import numpy as np
from tqdm import tqdm
import re
from scipy import stats
dataDir = '/u/cs401/A3/data/'


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    # n ← The number ofwords in REF
    n = len(r)
    # m ← Thenumberofwords in HYP
    m = len(h)
    # R ← zeros(n + 1, m + 1) // Matrixofdistances
    R = np.zeros((n + 1, m + 1))
    # B ← zeros(n + 1, m + 1) // Backtrackingmatrix
    B = np.zeros((n + 1, m + 1, 3))
    # Foralli, j s.t.i = 0 or j = 0, set R[i, j] ← ∞, except R[0, 0] ← 0
    R[0, :] = np.inf
    R[:, 0] = np.inf
    R[0, 0] = 0
    
    # for i = 1..n do
    for i in range(1, n + 1):
        #     for j = 1..m do
        for j in range(1, m + 1):
            #         del ← R[i − 1, j] + 1
            nDel = R[i - 1, j] + 1
            #         sub ← R[i − 1, j − 1] + (REF[i] == HY P[j])?0: 1
            sub = R[i - 1, j - 1] + (0 if r[i-1] == h[j-1] else 1)  # Sure? i and j?
            #         ins ← R[i, j − 1] + 1
            ins = R[i, j - 1] + 1
            #         R[i, j] ← Min(del, sub, ins )
            R[i, j] = min(nDel, sub, ins)
            #         if R[i, j] == del then
            if R[i, j] == nDel:
                #             B[i, j] ← ‘up’
                B[i, j, :] = B[i - 1, j, :]
                B[i, j, 0] += 1
            #         else if R[i, j] == ins then
            elif R[i, j] == ins:
                #             B[i, j] ← ‘left’
                B[i, j, :] = B[i, j - 1, :]
                B[i ,j ,1] += 1
            #         else
            else:
                #             B[i, j] ← ‘up - left’
                B[i, j, :] = B[i - 1, j - 1, :]
                B[i, j, 2] += (0 if r[i - 1] == h[j - 1] else 1)
    return np.round(R[-1, -1] / n, 4), B[-1, -1, 2], B[-1, -1, 1], B[-1, -1, 0]

def preprocess1(in_sentence):
    #lower-case and replace punc
    modComm = in_sentence.lower()
    modComm = re.sub(r"([\!\"\#\$\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\^\_\`\{\|\}\~])", " ", modComm)
    return modComm

def preprocess2(in_sentence):
    #remove multiple spaces, and broken to pieces
    
    return in_sentence.strip().split()
    
if __name__ == "__main__":
    print('TODO')
    trans_ref = None
    trans_google = None
    trans_kaldi = None
    fOut = open('asrDiscussion.txt', 'w')
    wer_Google = []
    wer_Kaldi = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in tqdm(dirs):
            transcript_path = os.path.join(dataDir, speaker)
            trans_ref = open(os.path.join(transcript_path, 'transcripts.txt'), 'r').read()
            trans_ref_lines = preprocess1(trans_ref).splitlines()
            trans_google = open(os.path.join(transcript_path, 'transcripts.Google.txt'), 'r').read()
            trans_google_lines = preprocess1(trans_google).splitlines()
            trans_kaldi = open(os.path.join(transcript_path, 'transcripts.Kaldi.txt'), 'r').read()
            trans_kaldi_lines = preprocess1(trans_kaldi).splitlines()
            len_ref = len(trans_ref_lines)
            len_google = len(trans_google_lines)
            len_kaldi = len(trans_kaldi_lines)

            if len_ref > 0 and len_google > 0 and len_kaldi > 0:
                num_lines = min(len_ref, len_google, len_kaldi)
                for i in range(num_lines):
                    line_ref = preprocess2(trans_ref_lines[i])
                    WER, S, I, D = Levenshtein(line_ref, preprocess2(trans_google_lines[i]))
                    print("{} {} {} {} S:{}, I:{}, D:{}".format(speaker,"Google", i, WER, S, I, D), file=fOut)
                    wer_Google.append(WER)
                    WER2, S2, I2, D2 = Levenshtein(line_ref, preprocess2(trans_kaldi_lines[i]))
                    print("{} {} {} {} S:{}, I:{}, D:{}".format(speaker,"Kaldi" , i, WER2, S2, I2, D2), file=fOut)
                    wer_Kaldi.append(WER2)
    print("Kaldi Avg: {} Google Avg: {} Kaldi std: {} Google std: {} ".format(np.mean(wer_Kaldi), np.mean(wer_Google), np.std(wer_Kaldi), np.std(wer_Google)), file=fOut)
    shapiro_google = stats.shapiro(wer_Google)
    shapiro_kaldi = stats.shapiro(wer_Kaldi)
    normality = ""
    if (shapiro_google[1] > 0.05 and shapiro_kaldi[1] > 0.05):
        normality = "Both Google({}) and Kaldi({}) marks are normal by shapiro test.".format(shapiro_google[1], shapiro_kaldi[1])
    else:
        normality = "At least one in Google({}) and Kaldi({}) marks are not normal by shapiro test. So test is not accurate.".format(shapiro_google[1], shapiro_kaldi[1])

    rst = stats.ttest_rel(wer_Google, wer_Kaldi)
    normality += "Paired t test, p-value {}".format(rst[1])
    print(normality)
        
                    

            
    fOut.close()
            
