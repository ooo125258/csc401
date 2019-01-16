import sys
import argparse
import os
import json
import re
import string

import html

indir = '/u/cs401/A1/data/';

abbrs = []
clitics = []

def init():
    global abbrs
    global clitics
    abbr_filename = "./abbrev.english"
    if not os.path.isfile(abbr_filename):
        abbr_filename = "/u/cs401/Wordlists/abbrev.english"
    with open(abbr_filename) as f:
        for line in f:
            abbrs.append(line.rstrip('\n'))

    clitics_filename = "./clitics"
    if not os.path.isfile(clitics_filename):
        clitics_filename = "/u/cs401/Wordlists/clitics"
    with open(clitics_filename) as f:
        for line in f:
            clitics.append(line.rstrip('\n'))

def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    #TODO: the problem of e.g....
    #TODO: e.g ....  alpha...
    comment = "&gt; You mean, besides being one of a handful of states known to be a State sponsor of terrorism?\
You mean like us in the e.g. United States supporting [Cuban terrorists](http://en.wikipedia.org/wiki/Luis_Posada_Carriles) or [Al-Qaeda](http://en.wikipedia.org/wiki/Al-Qaeda)!?? \
&gt;I wouldn't go so far as to say the Mr. Guardian Council... and Supreme Leader are rational - and given they are Islamists,?.?the concept of MAD does not apply.\
Really? Because they're Islamist they're not rational? That's why I don't like it.\n\nAny more! e.g.... alpha..."
    modComm = ''
    init()
    if 1 in steps:
        print('Remove all newline characters.')
        modComm = re.sub(r"(\.?)\n+", r" \1 ", comment)
        #modComm = comment.replace("\.\n", " \.")
        #modComm = comment.replace("\n", "")
    if 2 in steps:
        print('Replace HTML character codes with their ASCII equivalent.')
        modComm = html.unescape(modComm)
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
        modComm = re.sub(r"[\(\[\{]?http[s]?://[A-Za-z0-9\/\_\.\!\#\$\%\&\\\'\*\+\,\-\:\;\=\?\@\^\|\.]+[\)\]\}]?", '',
                         modComm)
        modComm = re.sub(r"[\(\[\{]?www\.[A-Za-z0-9\/\_\.\!\#\$\%\&\\\'\*\+\,\-\:\;\=\?\@\^\|]+[\)\]\}]?", '', modComm)
        print(1)
    if 4 in steps:
        # skip abbr, or others

        """
        What to replace here? We are using regex!
        ... need to be together
        ?!? need to be together
        Mr., e.g. need to be together
        others split
        
        So it's the only problem for . and ', besides the second problem
        """
        # Dot here is a problem. The one dot situation will be handled later. For all "..." and "?.?" will be handled
        new_modComm = re.sub(
            r"(\w|\s\^)([\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\.]{2,}|[\!\#\$\%\&\\\(\)\*\+\,\-\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]|\.\.\.)(\w|\s|$)",
            r"\1 \2 \3", modComm)

        # Handle the dot problem. If find a word followed by dot, then check if it's a word in abbrs list
        def periodHandler(matched):
            lst = abbrs
            word = matched.group().strip()
            if word in abbrs:
                return " " + word + " "
            else:
                return " " + word[:len(word) - 1] + " " + "." + " "
        #e.g.. , e.g. something, etc. something, something.
        #Another situation is something.\nsomething, such that it's connected!
        new_modComm = re.sub(r"(^|\s)((\w+\.)+\.?)($|\s)", periodHandler, new_modComm)
        modComm = new_modComm

        print('TODO')
    if 5 in steps:  # To this state, there is space before, then it's cite; there is no space, it's poss.
        # Notice, it's violated when skip step 4!

        # read from clitics
        # split '
        # merge back clitics


        print('TODO')
        def citeHandler(matched):
            lst = clitics
            sth_matched = str(matched.group())
            word = matched.group().strip()
            for i in range(len(clitics)):
                if clitics[i] in word:
                    ret = " " + word.replace(clitics[i], " " + clitics[i] + " ") + " "
                    return ret
            ret = word.replace("'", " ' ")
            return ret
        new_modComm = re.sub(r"(^|\s)(\w*\'\w*)($|\s)", citeHandler, modComm)
        modComm = new_modComm
    if 6 in steps:
        print('TODO')
    if 7 in steps:
        # split to list, remove if in stopwords
        print('TODO')
    if 8 in steps:#If 1 is executed, then the sentenceis split
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

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
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
