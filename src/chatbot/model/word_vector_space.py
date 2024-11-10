'''Module for the vector space LLM training '''

from gensim.models import Word2Vec

class WordVecModel:
    def __init__(self, model_params, corpus):
        self.model = None
        self.model_params = model_params
        self._sentences = self.get_sentences(corpus.get_doc_arr())


    def get_sentence(self, txt: str):
        '''convert string to array of words '''
        txt = txt.lower() 
        for i,c in enumerate(txt):
            if not c.isalpha():
                txt = txt[:i] + " " + txt[i+1:]
        tarr = [ s for s in txt.split(sep=' ') if len(s) > 0 ] 
        
        return tarr 

    def get_sentences(self, doc_arr):
        arr = [ self.get_sentence(s) for s in doc_arr ]

        print(arr)

        return arr

    # word2vec ---------------------------

    def train(self):
        vector_size, window, min_count, sg = self.model_params
        self.model = Word2Vec(self._sentences, 
                               vector_size=vector_size,
                               window = window, 
                               min_count = min_count, 
                               sg = sg 
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

 
 