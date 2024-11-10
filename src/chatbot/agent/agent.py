'''module for the agent object '''
from chatbot.preprocess.corpus import Corpus
from chatbot.model.model_factory import model_factory

class Agent:
    '''agent class that is an instance of a chatbot'''

    def __init__(self, name, corpus = None):
        self.name = name
        if corpus is None:
            corpus = Corpus() 
        self.corpus = corpus 
        self.models = {} 

    def __repr__(self):
        return f"My name is {self.name}. I have {len(self.corpus)} documents in my corpus."

    # Save and Load: -----------------------------
    def load(self):
        pass

    def save(self):
        pass

    # training and responding: -----------------------------
    def train(self, model_name, model_params, retrain=False):
        ''' train the model indicated with current corpus '''
        modelInit = model_factory.get_model(model_name)
        model = modelInit(model_params, self.corpus)
        model.train()
        self.models[model_name] = model

    def record_response(self, txt):
        '''record response given to the corpus'''
        self.corpus.add_text(txt)

    def clear_corpus(self):
        '''clear the corpus '''
        self.corpus.clear()

    def generate_response(self):
        '''generate a response based on the context '''
        pass