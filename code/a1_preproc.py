import sys
import argparse
import os
import json
import re
import string

import html
import spacy

indir = '/u/cs401/A1/data/';

abbrs = []
abbrs_lower = []
clitics = []
clitics_lower = []
stopwords = []
pn_abbrs = []
nlp = spacy.load('en', disable=['parser', 'ner'])

def init():
    global abbrs
    global abbrs_lower
    global clitics
    global clitics_lower
    global stopwords
    global pn_abbrs
    global nlp

    nlp = spacy.load('en', disable=['parser', 'ner'])
    if len(abbrs) == 0:
        abbr_filename = "./abbrev.english"
        if not os.path.isfile(abbr_filename):
            abbr_filename = "/u/cs401/Wordlists/abbrev.english"
        with open(abbr_filename) as f:
            for line in f:
                abbrs.append(line.rstrip('\n'))
                    
        abbrs_lower = [x.lower() for x in abbrs]

    
    if len(clitics) == 0:
        clitics_filename = "./clitics"
        if not os.path.isfile(clitics_filename):
            clitics_filename = "/u/cs401/Wordlists/clitics"
        with open(clitics_filename) as f:
            for line in f:
                clitics.append(line.rstrip('\n'))
        clitics_lower = [x.lower() for x in clitics]

    if len(stopwords) == 0:
        stopwords_filename = "./StopWords"
        if not os.path.isfile(stopwords_filename):
            stopwords_filename = "/u/cs401/Wordlists/StopWords"
        with open(stopwords_filename) as f:
            for line in f:
                stopwords.append(line.rstrip('\n'))
        # stopwords = set(stopwords)

    if len(pn_abbrs) == 0:
        pn_abbrs_filename = "./pn_abbrev.english"
        if not os.path.isfile(pn_abbrs_filename):
            pn_abbrs_filename = "/u/cs401/Wordlists/pn_abbrev.english"
        with open(pn_abbrs_filename) as f:
            for line in f:
                add_word = line.rstrip('\n')
                pn_abbrs.append(add_word)
                if add_word not in abbrs:
                    abbrs.append(add_word)
        # pn_abbrs = set(pn_abbrs)


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    #comment = "&gt; You mean, besides being one of a handful of states known to be a State sponsor of terrorism?\
    #You mean like us in the e.g. United States supporting [Cuban terrorists](http://en.wikipedia.org/wiki/Luis_Posada_Carriles) or [Al-Qaeda](http://en.wikipedia.org/wiki/Al-Qaeda)!?? \
    #&gt;I wouldn't go so far as to say the Mr. Guardian Council... and Supreme Leader are rational - and given they are Islamists,?.?the concept of MAD does not apply.\
    #Really? Because they're Islamist they're not rational? That's why I don't like it.\n\nAny more! e.g.... alpha... clitic's dogs'. dogs'  t'challa - y'all 'I don't think it's a good sentence.'"
    modComm = ''
    init()
    modComm_tags = []
    modComm_text = []
    modComm_lemma = []
    global nlp
    doc = None # doc is shared in one preproc1

    if 1 in steps:
        if comment == "":
            return ""
        #print('Remove all newline characters.')
        modComm = re.sub(r"(\.?)(\r?\n)+", r" \1 ", comment)
        # modComm = comment.replace("\.\n", " \.")
        # modComm = comment.replace("\n", "")
    if 2 in steps:
        if modComm == "":
            return ""
        #print('Replace HTML character codes with their ASCII equivalent.')
        modComm = re.sub(r"(\&\w+\;)", r" \1 ", modComm)#TODO: to add space around the HTML char. But DOES IT WORTH IT?
        modComm = html.unescape(modComm)

    if 3 in steps:
        if modComm == "":
            return ""
        #print('Remove all URLs http/www/https')
        #http://www.noah.org/wiki/RegEx_Python#URL_regex_pattern
        modComm = re.sub(r'[\(\[\{](?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+[\)\]\}]?', '', modComm)
            
    if 4 in steps:#TODO: modify if still slow. remember the capital abbr words
        # skip abbr, or others

        """
        What to replace here? We are using regex!
        ... need to be together
        ?!? need to be together
        Mr., e.g. need to be together
        others split
        
        So it's the only problem for . and ', besides the second problem
        """
        if modComm == "":
            return ""
        # Dot here is a problem. The one dot situation will be handled later. For all "..." and "?.?" will be handled
        # It's ridiculous, but ... has higher priority.
        new_modComm = re.sub(r"(\.\.\.)(\w|\s|$)", r" \1 \2", modComm)
        new_modComm = re.sub(
            r"(\w|\s\^)(\.\.\.|[\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\.\']{2,}|[\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~])(\w|\s|$)",
            r"\1 \2 \3", new_modComm)
        # Special operation for brackets
        new_modComm = re.sub(r"(\W|\^)([\[\(\{\'\"])", r"\1 \2 ", new_modComm)
        #quote is a problem. when \w+\', you don't know if it's person's or the player'FLASH' or sth.
        #But you are sure that if \s\', it must be a quote for reference!
        new_modComm = re.sub(r"(\]|\)|\})(\W|\$|\.)", r" \1 \2", new_modComm)
        # Handle the dot problem. If find a word followed by dot, then check if it's a word in abbrs list
        def periodHandler(matched):
            lst = abbrs_lower
            word = matched.group().strip()
            if word in lst:
                return " " + word + " "
            elif word[-1] == ' ' or word[-1] == '.':
                return " " + word[:len(word) - 1] + " " + "." + " "
            else:  # There is such a situation: apple.Tree e.g..Tree apple.E.g. So I choose the capital to be the identifier.
                return re.sub(r"(\w+)(\.)([A-Z])$", r" \1 \2 \3", word)

        # e.g.. , e.g. something, etc. something, something.
        # Another situation is something.\nsomething, such that it's connected!
        new_modComm = re.sub(r"(^|\s)((\w+\.)+\.?)($|\s|\w+)", periodHandler, new_modComm)
        # special case: dogs'. or t'.
        modComm = new_modComm.replace("'.", "' .")

    if 5 in steps:# split clitics
        if modComm == "":
            return ""
        def citeHandler(matched):
            lst = clitics_lower
            sth_matched = str(matched.group())
            word = matched.group().strip()
            for i in range(len(lst)):
                if clitics[i] in word:
                    ret = " " + word.replace(lst[i], " " + lst[i] + " ") + " "
                    return ret
            ret = word.replace("'", " ' ")
            return ret

        new_modComm = re.sub(r"(^|\s)(\w*\'\w*)($|\s)", citeHandler, modComm)
        modComm = new_modComm

    if 6 in steps:  #tags
        #We already know that 689 will be together.
        if modComm == "":
            return ""
        if nlp is None:
            print("warning: trying to load spacy in a wrong place. It would be much slower if it happens a lot of time!")
            nlp = spacy.load('en', disable=['parser', 'ner'])
        doc = spacy.tokens.Doc(nlp.vocab, words=modComm_lst)
        doc = nlp.tagger(doc)
        new_modComm_lst = []
        for token in doc:
            new_modComm_lst.append(token.text + "/" + token.tag_)
        modComm = " ".join(new_modComm_lst)

    if 7 in steps:  # Stop words
        # split to list, remove if in stopwords
        # analysis when it has or not the tags
        #replace beta/nng or beta/-lrc- or 
        #start and end with white space or head/end
        if modComm == "":
            return ""
        reStr = r"\b(" + r"(\s|^)" + r"|".join(stopwords) + r"\/\-?\S+\-?" + r"(\s|$)" + r")\b"
        modcComm = re.sub(reStr, " ", modComm, flags=re.IGNORECASE)
        new_modComm_lemma.append(modComm_lemma[idx])
        
        '''
        modComm_lst = modComm.split()
        if len(modComm_text) > 0:
            new_modComm_lst = []
            new_modComm_text = []
            new_modComm_tags = []
            new_modComm_lemma = []
            for idx in range(len(modComm_lst)):  # new_modComm_lst, modComm_text, modComm_tags have the same dimension
                if modComm_text[idx] not in stopwords:
                    new_modComm_lst.append(modComm_lst[idx])
                    new_modComm_text.append(modComm_text[idx])
                    new_modComm_tags.append(modComm_tags[idx])
                    new_modComm_lemma.append(modComm_lemma[idx])
            modComm_text = new_modComm_text
            modComm_tags = new_modComm_tags  # Save tags for later
            modComm_lemma = new_modComm_lemma
        else:
            new_modComm_lst = [x for x in modComm_lst if x.split('/')[0] not in stopwords]
        modComm = " ".join(new_modComm_lst)
        #print('TODO')
        '''

    if 8 in steps:  # lemmazation,
        #As piazza, 689 would be executed together.
        #So it must have tags
        #As we almost lemmazation all words, then it's actually useless to use regex here.
        #Warning! stopwords may or maynot be removed!
        
        if modComm == "":
            return ""

        modComm_lst = modComm.split()
        new_modComm = []
        '''
        
        Yes, the stopwords may or may not removed but it won't be reduced.
        If a word happens n times, it will still be n times or zero times.
        And the order of all words keep the same.
        So I use the word in modComm, scan from the left to the right
        '''
        j = 0
        for i in range(len(modComm_lst)):
            new_modComm_text, new_modComm_tag = token.split()
            while not new_modComm_text == doc[j]:
                j += 1
                #if j ends when i not, there would be a critical error!
            #Now I find the correct place
            token = doc[j]
            if token.lemma_[0] == '-' and token.text[0] != '-':  ##curcumstance to keep token
                new_modComm_lst.append(modComm_lst[i])#It doesn't change at all!
            elif token.lemma_[0] == token.text[0].lower():
                new_modComm_lst.append(token.text[j][0] + token.lemma_[j][1:] + "/" + new_modComm_tag)
                # Now we have word lists lemmaed.
            else:
                new_modComm_lst.append(token.lemma_[j] + "/" + new_modComm_tag)
            j += 1

            if j > len(doc):
                print("ERROR! j out of bound in step 8. Result bypass!")
                break
        modComm = " ".join(new_modComm_lst)

    if 9 in steps:  # new line between sentences 4.2.4.
        #print('TODO')  # Mr./NNP Guardian/NNP Council/NNP .../: \n and/CC Supreme/NNP
        #The code in comment was 4.2.4. However... as 689 together, we can use tags...
        if modComm == "":
            return ""

        modComm = re.sub(r"(\S\/\.)(\s|$)", "\1 \n\2", modComm)
        '''
        tag_scanner = r"\s*" #Handle tag exists or not
        if len(modComm_lst) == 0:
            print("ERROR here!")
        if len(modComm_lst[0].split('/')) == 2:  # Then there would be tags:
            tag_scanner = r"\/\S+\s*"
        #Place putative newline
        new_modComm = re.sub(r"([\.\?\!\;\:])(" + tag_scanner + r")", r"\1\2 \n ", modComm)
        #push newline after the following quotes
        new_modComm = re.sub(r"\n(\s*[\'\"])(" + tag_scanner + r")", r"\1\2 \n ", new_modComm)

        def abbrCheckForName(matched):
            word = matched.group().strip()
            word_lst = word.split()
            if len(word_lst) != 2:
                print("Step 9 Error in preprocess: request format: Prof. Rudzicz, actual word: " + word)
            if word_lst[0].split('/')[0] in pn_abbrs:
                return " " + word_lst[0] + " " + word_lst[1]
            elif ord(word_lst[1]) >=97 and ord(word_lst[1]) <= 122:
                return " " + word_lst[0] + " " + word_lst[1]
            else:
                return matched.group()
        #known abbr + . + Name/noneCapital?
        #Yes, I am speaking to you two: e.g. and i.e.
        new_modComm = re.sub(r"((\w+\.)+)" + r"(" + tag_scanner + r")" + r"\n(\s*\S)", abbrCheckForName, new_modComm)
        #Disqualify newline with ?! when following lowercase
        new_modComm = re.sub(r"([\?\!\:\;])" + r"(" + tag_scanner + r")" + r"\n(\s*^[A-Z])", r"\1\2\3", new_modComm)
        #Special case: ... word
        new_modComm = re.sub(r"(\s+...)" + r"(" + tag_scanner + r")" + r"\n(\s*[a-z])", r"\1\2\3", new_modComm)

        modComm = new_modComm
        '''
    if 10 in steps:  # lowercase
        if modComm == "":
            return ""
        modComm = modComm.lower()

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for fl in files:
            fullFile = os.path.join(subdir, fl)
            #print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            if args.max > 10000:
                args.max = 10000
            # TODO: read those lines with something like `j = json.loads(line)`
            startpt = args.ID[0] % len(data)
            mydata = data[startpt: startpt + args.max]
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            retained_mydata = []
            counter = 0
            for line in mydata:
                counter = counter + 1
                if counter % 100 == 0:
                    print("file {} counter {}".format(fl, counter))
                j = json.loads(line)
                newline = {}
                newline['id'] = j['id']
                newline['cat'] = fl

                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # TODO: replace the 'body' field with the processed text
                newline['body'] = preproc1(j['body'])
                retained_mydata.append(newline)
            # TODO: append the result to 'allOutput'

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
