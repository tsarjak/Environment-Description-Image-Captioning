# Dependencies
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from PIL import Image
from cache import cache
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# Loading the captions dataset from cache (pkl file)
import coco
_, filenames_train, captions_train = coco.load_records(train=True)


def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it to the given size if not None.
    """

    img = Image.open(path)

    # Resize image if desired.
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array.
    img = np.array(img)

    # Scaling the image
    img = img / 255.0

    # Convert 2D to 3D
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


# Loading the VGG16 image model (Autodownloads if not on computer)
image_model = VGG16(include_top=True, weights='imagenet')

# Fetching the fully-connected-2 layer from VGG16
transfer_layer = image_model.get_layer('fc2')

# Creating a new model from input to fc2
image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

# Getting the input size and output size for new transfer model
img_size = K.int_shape(image_model.input)[1:3]
transfer_values_size = K.int_shape(transfer_layer.output)[1]


# Start and End of caption sentence declaration
mark_start = 'ssss '
mark_end = ' eeee'

def mark_captions(captions_listlist):
    '''
    Mark the captions from list of lists with mark_start and mark_end
    '''

    captions_marked = [[mark_start + caption + mark_end for caption in captions_list] for captions_list in captions_listlist]
    return captions_marked

# Call the mark_captions function
captions_train_marked = mark_captions(captions_train)


def flatten(captions_listlist):
    captions_list = [caption for captions_list in captions_listlist for caption in captions_list]
    return captions_list

captions_train_flat = flatten(captions_train_marked)


# Total number of words we will be using for the captioning (picks the top 10k words from the training dataset)
num_words = 10000

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """
        
        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        
        return tokens


# Creating the tokenizer object
tokenizer = TokenizerWrap(texts=captions_train_flat, num_words=num_words)


# Getting the index of token_start and token_end (int values for ssss and eeee)
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]


# Creating the RNN model
state_size = 512
embedding_size = 128
num_words = 10000

transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')

decoder_transfer_map = Dense(state_size, activation='tanh', name='decoder_transfer_map')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3', return_sequences=True)
decoder_dense = Dense(num_words, activation='linear', name='decoder_output')


# Connecting the previously declared
def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches the internal state of the GRU layers. 
    # This means we can use the mapped transfer-values as the initial state of the GRU layers.
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.
    net = decoder_input
    
    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output


decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])



# Might not be needed for generating captions (But well)
def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    """

    # Calculate the loss. This outputs a 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean


optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_model.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[decoder_target])

# Load the weights
decoder_model.load_weights("trained_RNN.keras")


def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # VGG accepts images in 4D (batchsize, x, y, channels), hence, adding 1 to the batchsize
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = 2

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While maximum tokens reached or reached 'eeee'
    while token_int != 3 and count_tokens < max_tokens:
        # Update the input-sequence to the decoder with the last token that was sampled.
        # In the first iteration this will set the first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety, so we are sure we input the data in the right order.
        x_data = {'transfer_values_input': transfer_values, 'decoder_input': decoder_input_data}
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
    #plt.imshow(image)
    #plt.show()
    
    # Print the predicted caption.
    print("Predicted caption:")
    print(output_text[:-5])
    os.system("say " + output_text[:-5])
    print()