''' module to get the corpus '''
from sklearn.feature_extraction import text
from gensim.models import Word2Vec
from chatbot.preprocess.tokenstem import tokenize

class Corpus:
    '''corpus class that has the text and does the vectorization '''

    def __init__(self, text_arr:list =[] ):
        self._text_arr = text_arr
        self._sentences = self.get_sentences(text_arr) 

        # word2vec models 
        self.model = None  

        # vectorizers -- prevalance / frequency  
        self.vectorizer = text.CountVectorizer(binary=True)
        # self.vect = text.CountVectorizer(tokenizer=tokenize)
        self.vect = text.TfidfVectorizer(tokenizer=tokenize)
        return 

    def add_text(self, text):
        if isinstance(text, str):
            self._text_arr.append(text)
            # add to sentence 
            self._sentences +=  [ self.get_sentence(text) ] 
        elif isinstance(text, list) or isinstance(text,tuple):
            for t in text:
                self.add_text(t)
                # add to sentence 
                self._sentences +=  [ self.get_sentence(t) ]
 
        return 

    def get_sentence(self, txt: str):
        '''convert string to array of words '''
        txt = txt.lower() 
        for i,c in enumerate(txt):
            if not c.isalpha():
                txt = txt[:i] + " " + txt[i+1:]
        tarr = [ s for s in txt.split(sep=' ') if len(s) > 0 ] 
        
        return tarr 

    def get_sentences(self, text_arr):
        arr = [ self.get_sentence(s) for s in text_arr ]
        return arr

    # temp print funciton
    def print_tok(self):
        pass
    
    def print_corpus(self, n=20):
        for i, t in enumerate(self._text_arr):
            suffix = ""
            if len(t) > n: 
                suffix = " ..."
            print(t[:n] + suffix)
            print( "--- as sent ---: ", self._sentences[i] )

    # prevalence counters --------------------

    def vectorize(self):
        self.vectorizer.fit(self._text_arr)
        vt = self.vectorizer.transform(self._text_arr)

        return vt

    def tokenize(self):
        vec = self.vect.fit(self._text_arr) 
        return vec

    # word2vec ---------------------------

    def train(self):
        self.model = Word2Vec(self._sentences, 
                               vector_size=10,
                               window = 5, 
                               min_count = 1, 
                               sg = 1 
                    )
        
    def get_vec(self, word):
        if self.model:
            if word in self.model.wv:
                return self.model.wv[word]
        
        return None

    def get_similarity(self,p,q):
        if self.model:
            if p in self.model.wv and q in self.model.wv:
                return self.model.wv.similarity(p,q)
        
        return None

 
    