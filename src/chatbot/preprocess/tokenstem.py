import nltk
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# from spellchecker import SpellChecker

# spell = SpellChecker()
stemmer = PorterStemmer()
stop_words = stopwords.words('english')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # Tokenizing
    tokens = word_tokenize(text.lower())

    # # apply spell checker
    # tokens = [ spell.correction(word) for word in tokens if word.isalpha() ]

    # Removing stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stems = stem_tokens(tokens, stemmer)
    return stems