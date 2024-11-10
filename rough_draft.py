from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
import numpy as np 

from chatbot.tests.sources.example_sources import iamgroot, spot

texts = iamgroot
seed_text = "I am Groot. I am Groot."

# texts = spot 
# seed_text = "I see spot"

# ================ tokenize that data -------

tokenizer = Tokenizer(oov_token="<OOV>", filters='', lower=False) 
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

# View results
print("Word Index:", tokenizer.word_index)
print("Sequences:", sequences)
print("Padded Sequences:", padded_sequences)


# Word2Vec model -----------------

index_word = {v: k for k, v in tokenizer.word_index.items() }

# create the tokenized sentences 
tokenized_sentences = [[index_word[token] for token in seq if token!=0] for seq in padded_sequences]

print("\n")
print("Original Sentences: ", texts)
print("Tokenized Sentences for Word2Vec:", tokenized_sentences)

word2vec_model = Word2Vec(sentences=tokenized_sentences, 
                               vector_size=30,
                               window = 3, 
                               min_count = 1, 
                               workers=4, 
                    )
# word2vec_model = Word2Vec.load('path_to_your_model')  # Load your model

# print('done')
# for word in ['cat', 'dog', 'dogs']:
#     print(word, word2vec_model.wv[word])


# embed into LSTM 
# embedding_matrix = word2vec_model.wv.vectors
# print("\nEmbedding matrix size: ", embedding_matrix.shape )

print( "Let's look at the word2vec model! These are different form the tokenizer!")

print("\n\nwv: ", word2vec_model.wv)
print("\n\nvocab: ", word2vec_model.wv.key_to_index)
print("\n\nvectors[0]: ", word2vec_model.wv.vectors[0] )
# print("\n\n vector for the index 0: ", word2vec_model.wv["the"])

# deal with the padding token to recreate the embedding matrix 
vocab_size = len(tokenizer.word_index)  + 1     # padding token 
embedding_dim = word2vec_model.vector_size 

embedding_matrix = np.zeros((vocab_size, embedding_dim))

# populate 
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

print("\nEmbedding matrix size: ", embedding_matrix.shape )
print("\nAnd the words are ... ")
for word, i in tokenizer.word_index.items():
    vec = None
    if word in word2vec_model.wv:
        vec = word2vec_model.wv[word]
    print(word, i, "\n", embedding_matrix[i], "\n", vec )

# Build the LSTM model
# max_sequence_length = max([len(seq) for seq in padded_sequences])

print("Vocab size= ", vocab_size)
print("Embedding dim: ", embedding_dim)
print("Max sequence length: ", max_sequence_length)

model = Sequential([
    Input(shape=(max_sequence_length,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    LSTM(64, dropout=0.2),
    Dense(vocab_size, activation='softmax')  # output layer 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Prepare input and output sequences
X = padded_sequences[:, :-1]  # All but the last word
y = padded_sequences[:, -1:]    # the last word

# make y categorical 
y = to_categorical(y, num_classes=vocab_size)


print("\nPadded sequences: \n", padded_sequences)

print("\nThis is X:\n", X)
print("\nThis is y:\n", y)

print("X shape: ", X.shape)
print("y shape: ", y.shape)

# get the model size 
print( "Model input: ", model.input_shape)
print( "Model output: ", model.output_shape )


model.fit(X, y, epochs=120, batch_size=32)  # Adjust epochs and batch size as needed


print(model.summary() )

def generate_text(model, tokenizer, seed_text, max_sequence_len, n_words=10):
    # Tokenize the seed text
    sequence = tokenizer.texts_to_sequences([seed_text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')

    # Generate text one word at a time
    predicted_text = seed_text

    for _ in range(n_words):

        # Predict the next word
        pred = model.predict(sequence, verbose=0)
        predicted_word_index = np.argmax(pred)  # Get the index of the predicted word

        # Reverse the index to word
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        
        # Append predicted word to the text
        predicted_text += ' ' + predicted_word

        # Update the sequence with the predicted word
        sequence = np.append(sequence, predicted_word_index)
        sequence = sequence[1:].reshape(1, -1)

    return predicted_text

# Example usage
# seed_text = "The cat"
generated_text = generate_text(model, tokenizer, seed_text, max_sequence_length, n_words=100)
print("Generated Text:", generated_text)

generated_text = generate_text(model, tokenizer, "Elephant. Elephant", max_sequence_length, n_words=100)
print("Generated Text:", generated_text)



exit() 



# # #  # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


