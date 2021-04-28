import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import (Dense, Conv2DTranspose, Reshape, Input,
                                     BatchNormalization, Conv2D, Flatten,
                                     LeakyReLU, Dropout)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class DCGAN:

    def __init__(self, loss=BinaryCrossentropy(from_logits=True),
                 path_save="DCGAN"):

        # parameters
        self.path_save = path_save
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100  # will be split in 2 parts (if we want)
        self.discri_optimizer = Adam(1e-4)
        self.stacked_optimizer = Adam(1e-4)
        self.loss = loss

        # discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=self.loss,
            optimizer=self.discri_optimizer,
            metrics=['accuracy'])

        # stacked
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        is_valid = self.discriminator(img)
        self.stacked = Model(z, is_valid)
        self.stacked.compile(loss=self.loss,
                             optimizer=self.stacked_optimizer,
                             metrics=['accuracy'])

    def build_generator(self):

        input = Input(shape=(self.latent_dim,))
        x = Dense(256 * 7 * 7)(input)
        x = Reshape((7, 7, 256))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(128, kernel_size=(5, 5),
                            padding='same', strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(64, kernel_size=(5, 5),
                            padding='same', strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                 activation='tanh')(x)

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

        x = Flatten()(x)
        output = Dense(1)(x)

        return Model(input, output)

    def train(self, epochs, batch_size=64):

        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train.astype(np.float32) / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)

        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))

        training_loss_d = []
        training_loss_g = []

        for epoch in range(epochs):

            print("images beginning epoch: ", epoch)
            self.sample_images()

            for i in range(0, len(X_train) - batch_size, batch_size):

                imgs = X_train[i: i + batch_size]

                # smarter noise generation
                noise = tf.random.normal(shape=(batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * tf.add(d_loss_fake, d_loss_real)
                # train Generator
                g_loss = self.stacked.train_on_batch(noise, valid)

            self.sample_images(epoch)
            training_loss_d.append(d_loss[0].numpy())
            training_loss_g.append(g_loss[0])

        return training_loss_d, training_loss_g

    def sample_images(self, epoch=None):
        z = tf.random.normal((9, self.latent_dim))
        gen_imgs = self.generator.predict(z)

        # Rescale images 0 - 1
        gen_imgs = (gen_imgs + 1) / 2

        fig, axs = plt.subplots(3, 3)
        cnt = 0
        for i in range(3):
            for j in range(3):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        if epoch is not None:
            fig.savefig(self.path_save + "_images_epoch_" + str(epoch))
        plt.show()
