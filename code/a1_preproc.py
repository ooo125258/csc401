import sys
import argparse
import os
import json
import re
import string

indir = '/u/cs401/A1/data/';

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        print('Remove all newline characters.')
        modComm = comment.replace("\n", "")
    if 2 in steps:
        print('Replace HTML character codes with their ASCII equivalent.')
        htmlCodes = (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        )
        for code in htmlCodes:
            modComm = modComm.replace(code[1], code[0])
    if 3 in steps:
        print('Remove all URLs http/www/https')
        modComm = re.sub("[\(\[\{]?http[s]?://[A-Za-z0-9\/\_\.\!\#\$\%\&\\\'\*\+\,\-\:\;\=\?\@\^\|]+[\)\]\}]?", '', modComm)
        modComm = re.sub("[\(\[\{]?www\.[A-Za-z0-9\/\_\.\!\#\$\%\&\\\'\*\+\,\-\:\;\=\?\@\^\|]+[\)\]\}]?", '', modComm)
        print(1)
    if 4 in steps:
        #skip abbr, or others
        abbrs = []
        with open("/u/cs401/Wordlists/abbrev.english") as f:
            for line in f:
                abbrs.append(line.rstrip('\n'))

        #split except . or '
        #merge if punc\spunc
        #split by ...
        #split to lst by space.
        #for each word with ., find the range of abbr. All . appear at other places will be split
        puncs = string.punctuation
        puncs = puncs.replace('.','').replace("'",'')

        new_modComm = ""
        prev = ""
        for i in modComm:
            if i in puncs:
                prev += i
            else:
                if prev == "":
                    new_modComm += i
                else:
                    new_modComm += ' ' + prev + ' ' + i
                    prev = ""

        if prev != "":
            new_modComm += ' ' + prev
        # split by ...
        new_modComm.replace('...', ' ... ')
        # split to lst by space.
        modComm_lst = new_modComm.rsplit("\s+")
        new_modComm_lst = []
        for word in modComm_lst:
            word_pos_occupied_by_abbr = [False] * len(modComm_lst)
            #check pos for abbr
            for abbr in abbrs:
                if abbr in word: #need to modify to correct py code
                    idxs = re.search(word, abbr)#TODO: It may return multiple result
                    for idx in idxs:
                        if idx + len(abbr) > len(modComm_lst):
                            print("Self-defined ERROR at preprocess Step4: the word {} finds abbr {} at index {}.".format(word, abbr, idx))
                        word_pos_occupied_by_abbr[idx : idx + len(abbr)] = [True] * len(abbr)
            #then check period outside abbr
            word_modified = None
            for i in range(len(word)):
                if word[i] == '.' and not word_pos_occupied_by_abbr[i]:
                    if word_modified is None:
                        word_modified = word[:i] + ' ' + '.' + ' '
                    else:
                        word_modified += ' ' + '.' + ' '
                else:
                    if word_modified is not None:
                        word_modified += word[i]
            if word_modified is None:
                new_modComm_lst.append(word_modified)
            else:
                new_modComm_lst.append(word)
            #join back the modComm
            modComm = "".join(new_modComm_lst)
        print(1)


        '''
        

        for code in skip_dict:
            modComm = modComm.replace(code[1], code[0])
        '''

        print('TODO')
    if 5 in steps:#To this state, there is space before, then it's cite; there is no space, it's poss.
        #Notice, it's violated when skip step 4!

        #read from clitics
        #split '
        #merge back clitics
        print('TODO')

        word_pos_occupied_by_cites
        new_modComm = ""
        prev = ""
        for i in modComm:
            if i == "'" and prev != :
                new_modComm += i
            else:
                new_modComm += " " + i + " "
            prev = i
    if 6 in steps:
        print('TODO')
    if 7 in steps:
        #split to list, remove if in stopwords
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            if args.max > 10000:
                args.max = 10000
            # TODO: read those lines with something like `j = json.loads(line)`
            startpt = args.ID[0] % len(data)
            mydata = data[startpt : startpt + args.max]
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            retained_mydata = []
            for line in mydata:
                j = json.loads(line)
                newline = {}
                newline['id'] = j['id']
                newline['cat'] = file

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
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
