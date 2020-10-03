import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


import spacy
nlp = spacy.load("en_core_web_md")

def prettify (pos_tagged):    
    p_2 = ""
    for z in pos_tagged:
        if (z[0].lower() != 'i') and (('VB' in z[1]) or ('NN' in z[1])):
            p_2= p_2 + " " + str(z[0].lower())
    if len(p_2) == 0:
        for z in pos_tagged:
            if len(z[0]) > 4 and (z[0]) != ' none':
                p_2= p_2 + " " + str(z[0].lower())
    return(p_2)

class FancyString(str):
    '''
    Extends string class for a variety of basic NLP tasks
            
    '''
    
    def __init__(self, sent):
        self.txt = sent
        
    def is_passive(self):
        z = nltk.word_tokenize(self)
        meh = nltk.pos_tag(z)
        f = prettify(meh)
        f = nltk.word_tokenize(f)
        f = nltk.pos_tag(f)
        q = 0
        countpass = 0
        passlist = []
        for n in f:
           # print(n)
            if (n[1] == 'VBN'):
                q = f.index(n)
                for z in meh[q-1]:
                    if ('have' not in meh[q-1] and 'has' not in meh[q-1] and 'had' not in meh[q-1] and 'VB' in z):
                        if ((meh[q-1][0] + ' ' + n[0]) not in passlist):
                            self.passive = 1
                            return(self.passive)
        self.passive = 0
        return(self.passive)

    def wordcount(self):
        split = self.split()
        self.length = (len(split))
        return(self.length)

    def me_count (self):    
        p_2 = ""
        split = self.split()
        first_single = ['i', 'me', 'mine', 'myself']
        first_s = 0
        for z in split:
            if (z.lower() in first_single):
                first_s += 1
        self.me_count = first_s
        return(self.me_count)

    def ad_count(self):
        z = nltk.word_tokenize(self)
        meh = nltk.pos_tag(z)
        adj = 0
        for i in meh:
            if (i[1] == 'RB') or (i[1] == 'JJ'):
                adj += 1
        self.ad_count = adj
        return(self.ad_count)
    
    def spacy_vec(self):
        v = str(self)
        v = nlp(v).vector
        self.spacy_vec = v
        return(self.spacy_vec)


