''' module to get the corpus '''
from sklearn.feature_extraction import text

from chatbot.preprocess.tokenstem import tokenize

class Corpus:
    '''corpus class that has the text and does the vectorization '''

    def __init__(self, doc_arr:list=None ):
        if doc_arr is None:
            doc_arr = []
        self._doc_arr = doc_arr
        # self._sentences = self.get_sentences(doc_arr) 

        # # word2vec models 
        # self.model = None  
        # # vectorizers -- prevalance / frequency  
        # self.vectorizer = text.CountVectorizer(binary=True)
        # # self.vect = text.CountVectorizer(tokenizer=tokenize)
        # self.vect = text.TfidfVectorizer(tokenizer=tokenize)
        return 

    def __len__(self):
        '''return the number of documents'''
        return len(self._doc_arr)

    def clear(self):
        self._doc_arr = [] 

    def get_doc_arr(self):
        return self._doc_arr

    def add_text(self, text):
        if isinstance(text, str):
            self._doc_arr.append(text)
            # add to sentence 
            # self._sentences +=  [ self.get_sentence(text) ] 
        elif isinstance(text, list) or isinstance(text,tuple):
            for t in text:
                self.add_text(t)
                # add to sentence 
                # self._sentences +=  [ self.get_sentence(t) ]
 
        return 

    # temp print funciton
    def print_tok(self):
        pass
    
    def print_corpus(self, n=20):
        for i, t in enumerate(self._doc_arr):
            suffix = ""
            if len(t) > n: 
                suffix = " ..."
            print(t[:n] + suffix)
            print( "--- as sent ---: ", self._sentences[i] )

    # prevalence counters --------------------

    def vectorize(self):
        self.vectorizer.fit(self._doc_arr)
        vt = self.vectorizer.transform(self._doc_arr)

        return vt

    def tokenize(self):
        vec = self.vect.fit(self._doc_arr) 
        return vec

   