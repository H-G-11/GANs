import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Dense, Conv2DTranspose, Reshape, Input,
                                     BatchNormalization, Conv2D, Flatten,
                                     LeakyReLU, Dropout)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop


class WGAN:

    def __init__(self, optimizer=RMSprop(learning_rate=9e-5)):

        # parameters
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100  # will be split in 2 parts (if we want)
        self.n_critic = 5
        self.clip_value = 0.01
        self.discri_optimizer = RMSprop(learning_rate=9e-5)
        self.stacked_optimizer = RMSprop(learning_rate=9e-5)

        # generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # stacked
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        is_valid = self.discriminator(img)
        self.stacked = Model(z, is_valid)
        self.stacked.compile(loss=self.wasserstein_loss,
                             optimizer=self.stacked_optimizer)

    def build_generator(self):

        input = Input(shape=(self.latent_dim,))
        x = Dense(256 * 7 * 7)(input)
        x = Reshape((7, 7, 256))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(128, kernel_size=(5, 5), use_bias=False,
                            padding='same', strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(128, kernel_size=(5, 5), use_bias=False,
                            padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                 use_bias=False, activation='tanh')(x)

        return Model(input, output)

    def build_discriminator(self):

        input = Input(shape=self.img_shape)
        x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                   padding='same')(input)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        output = Dense(1)(x)

        discriminator = Model(input, output)
        discriminator.compile(
            loss=self.wasserstein_loss,
            optimizer=self.discri_optimizer)

        return discriminator

    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

    def train(self, iterations, batch_size=64, print_every=1000):

        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train.astype(np.float32) / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)

        valid = -1 * tf.ones((batch_size, 1))
        fake = tf.ones((batch_size, 1))

        for iteration in range(iterations):

            for _ in range(self.n_critic):  # train discriminator

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # smarter noise generation
                noise = tf.random.normal(shape=(batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = tf.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for _l in self.discriminator.layers:
                    weights = _l.get_weights()
                    weights = [tf.clip_by_value(w, -self.clip_value,
                                                self.clip_value)
                               for w in weights]
                    _l.set_weights(weights)

            # train Generator
            g_loss = self.stacked.train_on_batch(noise, valid)
            if iteration % print_every == 0:  # plot the progress
                print(d_loss.numpy(), g_loss)
                self.sample_image()

    def sample_image(self):
        noise = tf.random.normal(shape=(9, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs + 1) / 2

        fig, axs = plt.subplots(3, 3)
        cnt = 0
        for i in range(3):
            for j in range(3):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
