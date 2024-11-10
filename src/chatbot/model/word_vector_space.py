'''Module for the vector space LLM training '''

from gensim.models import Word2Vec

class WordVecModel:
    def __init__(self, model_params, corpus):
        self.model = None
        self.model_params = model_params
        self.corpus = corpus 

        self._sentences = self.get_sentences()

    # preprocessing done by corpus 
    def get_sentences(self):
        index_word = {v: k for k, v in self.corpus.tokenizer.word_index.items() }

        # print("\nI am in get sentences...\n")
        # print(index_word)
        # self.corpus.update_texts() 
        # print(self.corpus.texts)
        # print(self.corpus.get_seq() )

        # create the tokenized sentences 
        tokenized_sentences = [[index_word[token] for token in seq if token!=0] for seq in self.corpus.get_seq()]

        return tokenized_sentences


    # word2vec ---------------------------

    def train(self):
        vector_size, window, min_count, nworkers, epochs = self.model_params
        sentences = self.get_sentences()
        self.model = Word2Vec(sentences, 
                               vector_size=vector_size,
                               window = window, 
                               min_count = min_count, 
                               workers = nworkers, 
                               epochs=epochs,
                               alpha=0.1,
                    )

    def update(self):
        new_sentences = self.get_sentences()
        self.model.build_vocab(new_sentences, update=True)
        self.model.train(new_sentences, total_examples=len(new_sentences), epochs=self.model.epochs)

    # Access the model  -----------------------------------

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

 
    # def get_sentence(self, txt: str):
    #     '''convert string to array of words '''
    #     txt = txt.lower() 
    #     for i,c in enumerate(txt):
    #         if not c.isalpha():
    #             txt = txt[:i] + " " + txt[i+1:]
    #     tarr = [ s for s in txt.split(sep=' ') if len(s) > 0 ] 
    #     
    #     return tarr 

    # def get_sentences(self, doc_arr):
    #     arr = [ self.get_sentence(s) for s in doc_arr ]

    #     print(arr)

    #     return arr

