''' module to get the corpus '''
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  

# cycle through so that each word is at end  
def cyclic_permutation(arr):
    if isinstance(arr, np.ndarray):
        return np.concatenate([arr[-1:], arr[:-1] ] ) 
    else:
        return arr[-1:] + arr[:-1]


def cycle_through(arr,nrepeat=1):
    ncycle = len(arr) * nrepeat
    out_arr = [ [] for i in range(ncycle) ] 
    for i in range(ncycle):
        out_arr[i] = arr
        arr = cyclic_permutation(arr)

    return out_arr

def full_cyclic_permutation(seq_arr, nrepeat=1):
    return [x for row in seq_arr for x in cycle_through(row, nrepeat)]



class Corpus:
    '''corpus class that has the text and does the vectorization '''

    def __init__(self, doc_arr:list=None ):
        if doc_arr is None:
            doc_arr = []
        self._doc_arr = doc_arr

        # use tf keras tokenizer
        self.tokenizer = self.get_tokenizer()  
        self.update_texts()  # updates self.texts 

        return 

    # Manage the doc array -----------------------------

    def __len__(self):
        '''return the number of documents'''
        return len(self._doc_arr)

    def clear(self):
        self._doc_arr = [] 
        self.texts = None

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

    # get texts (processed doc array) --------
    def update_texts(self):
        self.texts = self._doc_arr
        self.update_tokenizer() 

    def get_tokenizer(self):
        return Tokenizer(oov_token="<OOV>", filters='', lower=False) 
 
    def update_tokenizer(self):
        # Tokenize new texts
        new_tokenizer = self.get_tokenizer() 
        new_tokenizer.fit_on_texts(self.texts)
        new_word_index = new_tokenizer.word_index

        # Update the original tokenizer
        current_max_index = 0 
        if len(self.tokenizer.word_index.values() ) > 0: 
            current_max_index = max(self.tokenizer.word_index.values())
        for word, idx in new_word_index.items():
            if word not in self.tokenizer.word_index:
                current_max_index += 1
                self.tokenizer.word_index[word] = current_max_index
                self.tokenizer.index_word[current_max_index] = word

    def get_word_index(self):
        return self.tokenizer.word_index


    # token preprocessing -----------------------------

    def get_seq(self, *args, **kwargs):
        self.update_texts() 
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        
        return sequences

    def pad_seq(self, sequences=None, *args, **kwargs):
        if sequences is None:
            sequences = self.get_seq() 

        # Pad sequences
        max_sequence_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

        return padded_sequences


    def get_cyclic_perm(self, nrepeat=1):
        '''get cyclic permutation of the text to end on all words'''
        sequences = self.get_seq() 
        sequences = full_cyclic_permutation(sequences) 
        padded_seq = self.pad_seq(sequences)

        return padded_seq











# # # # # # # # # ## # # # ## # # # #
# from sklearn.feature_extraction import text
# from chatbot.preprocess.tokenstem import tokenize



        # # word2vec models 
        # self.model = None  
        # # vectorizers -- prevalance / frequency  
        # self.vectorizer = text.CountVectorizer(binary=True)
        # # self.vect = text.CountVectorizer(tokenizer=tokenize)
        # self.vect = text.TfidfVectorizer(tokenizer=tokenize)
 

    # def print_corpus(self, n=20):
    #     for i, t in enumerate(self._doc_arr):
    #         suffix = ""
    #         if len(t) > n: 
    #             suffix = " ..."
    #         print(t[:n] + suffix)
    #         print( "--- as sent ---: ", self._sentences[i] )

    # # prevalence counters --------------------
    # def vectorize(self):
    #     self.vectorizer.fit(self._doc_arr)
    #     vt = self.vectorizer.transform(self._doc_arr)

    #     return vt

    # def tokenize(self):
    #     vec = self.vect.fit(self._doc_arr) 
    #     return vec

   