import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import requests


def solution_model():
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    #datastore = urllib.request.urlretrieve(url, 'sarcasm.json')
    datastore = requests.get('http://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json')
    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    for item in datastore.json():
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    num_epochs = 10
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=2)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    #print(decode_sentence(training_padded[0]))
    #print(training_sentences[2])
    #print(labels[2])
    e = model.layers[0]
    weights = e.get_weights()[0]
    #print(weights.shape)  # shape: (vocab_size, embedding_dim)
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    model = solution_model()
    #model.summary()
    tokenizer = Tokenizer(oov_token="<OOV>")
    #sentence = "facebook reportedly working on healthcare features and apps"
    #sentence = "sentence = I really think this is amazing. honest."


    sentence = ["obama visits arlington national cemetery to honor veterans",
            "why writers must plan to be surprise",
            "gillian jacobs on what it's like to kiss adam brody",
            "rescuers heroically help beached garbage back into ocean",
            "christian bale visits sikh temple victims",
            "brita-unveils-new-in-throat-water-filters"]
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)


    #sequence = tokenizer.texts_to_sequences([sentence])
    #print(sequence)


    #sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    #print(padded)
    print(model.predict(padded))
    #model.save("sarcasm.h5")
