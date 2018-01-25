from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, UpSampling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

def generative_model(input_shape=(64, 64, 3), leaky_relu_alpha=0.2):
    lrelu = LeakyReLU(leaky_relu_alpha)

    model_input = Input(input_shape)

    # encoder
    x = Conv2D(16, (3, 3), activation=lrelu, padding='same')(model_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation=lrelu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation=lrelu, padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # decoder
    x = Conv2D(8, (3, 3), activation=lrelu, padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation=lrelu, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation=lrelu)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

    # Compile
    autoencoder = Model(model_input, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder