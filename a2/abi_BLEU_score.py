import math

def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    precisions = []
    # for each n-gram
    for i in range(1, n+1):
        c_words = candidate.split()
        nume = 0
        deno = len(c_words)
        # update nume
        for j in range(len(c_words)):
            if j < len(c_words) - (i-1):
                ngram = c_words[j:j+i]
                phrase = ' '.join(ngram)
                refs = ' '.join(references)
                if i == 1:
                    refs = refs.split()
                if phrase in refs:
                    nume += 1
        precisions.append(nume/deno)

    # finding the length of the reference with the closest length to candidate
    closest = 0
    for ref in references:
        if abs(len(ref) - len(candidate)) < abs(len(ref) - closest):
            closest = len(ref)

    brevity = closest / len(candidate)

    BP = 0
    if brevity < 1:
        BP = 1
    elif brevity >= 1:
        BP = math.exp(1 - brevity)

    precisions_calc = 0
    for i in precisions:
        if precisions_calc == 0:
            precisions_calc = i
        else:
            precisions_calc = precisions_calc * i
    
    bleu_score = BP * (precisions_calc**(1/n))

    return bleu_score

if __name__ == "__main__":
    candidate = "SENTSTART i am hungry SENTEND"
    references = ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n = 3
    blue  = BLEU_score(candidate, references, n)
    print(blue)
