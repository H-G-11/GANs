import h5py

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (Dense, Conv2DTranspose, Input,
                                     Conv2D, Flatten, LeakyReLU, ReLU, add)
from tensorflow.keras import Model


def hdf5(path, data_key="data", target_key="target"):
    """ loads data from hdf5:
    - hdf5 should have 'train' and 'test' groups
    - each group should have 'data' and 'target' dataset or spcify the key
    - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        X_tr = X_tr.reshape((len(X_tr), 16, 16))
        X_te = X_te.reshape((len(X_te), 16, 16))
    return X_tr, y_tr, X_te, y_te


def residual_block(input, activation=ReLU(), kernel_size=(3, 3),
                   strides=(1, 1)):
    """ I prefer to use convolution with same padding
    to avoid using upsampling. """

    filters = input.shape[-1]
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
               padding="same")(input)
    x = InstanceNormalization()(x)
    x = activation(x)

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
               padding="same")(input)
    x = InstanceNormalization()(x)
    # x = activation(x)  # to add or not to add ? This is the question

    x = add([x, input])

    return x


def build_discriminator(input_mnist=True):

    input_shape = (28, 28, 1) if input_mnist else (16, 16, 1)
    input = Input(input_shape)

    filters = 32
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(input)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    for _ in range(2):
        filters *= 2
        x = Conv2D(filters=filters, kernel_size=(2, 2), strides=(2, 2))(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Conv2D(8, kernel_size=(2, 2), strides=(1, 1))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    output = Dense(1)(x)
    return Model(input, output)


def generator_from_mnist_simplified(nb_residual_blocks=0):
    img_input = Input((28, 28, 1))

    filters = 16

    # x = Conv2D(filters, kernel_size=(4, 4), padding="same")(img_input)
    # x = InstanceNormalization()(x)
    # x = ReLU()(x)

    x = Conv2D(filters, kernel_size=(3, 3))(img_input)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    for _ in range(5):
        filters *= 2
        x = Conv2D(filters, kernel_size=(3, 3))(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)

    for _ in range(nb_residual_blocks):
        x = residual_block(x)

    output = Conv2D(1, (6, 6), activation='tanh', padding="same")(x)
    generator_mnist = Model(img_input, output)
    # generator_mnist.summary()
    return generator_mnist


def generator_from_usps_simplified(nb_residual_blocks=0):
    img_input = Input((16, 16, 1))

    filters = 16

    # x = Conv2D(filters, kernel_size=(4, 4), padding="same")(img_input)
    # x = InstanceNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(filters, kernel_size=(3, 3))(img_input)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    for _ in range(5):
        filters *= 2
        x = Conv2DTranspose(filters, kernel_size=(3, 3))(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)

    for _ in range(nb_residual_blocks):
        x = residual_block(x)

    output = Conv2D(1, (2, 2), activation='tanh', padding="same")(x)
    generator_usps = Model(img_input, output)
    # generator_usps.summary()
    return generator_usps
