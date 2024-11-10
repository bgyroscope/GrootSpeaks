'''module for the model factory '''
from chatbot.model.word_vector_space import WordVecModel

class modelBuilder:
    ''' handles the acces to '''
    pass

class modelFactory:
    '''a class to initiate various models '''

    def __init__(self):
        self._creators = {} 

    def register_model(self, format, creator):
        self._creators[format] = creator 

    def get_model(self, format):
        creator = self._creators.get(format)
        if not creator:
            raise ValueError(format)
        return creator 


model_factory = modelFactory() 
model_factory.register_model("WORD2VEC", WordVecModel)