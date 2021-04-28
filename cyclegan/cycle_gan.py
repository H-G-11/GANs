import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

from .utils import (generator_from_mnist_simplified, hdf5,
                    generator_from_usps_simplified,
                    build_discriminator)


class CycleGAN:

    def __init__(self, lambda_cycle=10):
        self.lambda_cycle = lambda_cycle
        # models
        self.generator_from_m = generator_from_mnist_simplified(1)
        self.generator_from_u = generator_from_usps_simplified(1)
        self.discriminator_from_m = build_discriminator()
        self.discriminator_from_u = build_discriminator(input_mnist=False)
        # optimizers
        self.generator_from_m_opti = Adam(learning_rate=3e-4, beta_1=0.5)
        self.generator_from_u_opti = Adam(learning_rate=3e-4, beta_1=0.5)
        self.discriminator_from_m_opti = Adam(learning_rate=3e-4, beta_1=0.5)
        self.discriminator_from_u_opti = Adam(learning_rate=3e-4, beta_1=0.5)
        # load data
        self._load_data()

    def _load_data(self):
        (X_mnist, _), (_, _) = mnist.load_data()
        X_usps, _, _, _ = hdf5('GAN/data/usps.h5')
        X_mnist = X_mnist.astype(np.float32) / 127.5 - 1
        X_usps = X_usps.astype(np.float32) * 2 - 1
        self.X_mnist = np.expand_dims(X_mnist, axis=3)
        self.X_usps = np.expand_dims(X_usps, axis=3)

    def train(self, iterations=1000, batch_size=64, verbose=1,
              frequence_sample=10):

        for iter in range(iterations):
            idx_mnist = np.random.randint(0, len(self.X_mnist), batch_size)
            img_mnist = self.X_mnist[idx_mnist]
            idx_usps = np.random.randint(0, len(self.X_usps), batch_size)
            img_usps = self.X_usps[idx_usps]

            # ===============================
            # ==== update discriminators ====
            # ===============================
            with tf.GradientTape(persistent=True) as tape_discri:
                # on real
                loss_discri_mnist_real = tf.reduce_mean(
                    (self.discriminator_from_m(
                        img_mnist, training=True) - 1) ** 2, axis=0)
                loss_discri_usps_real = tf.reduce_mean(
                    (self.discriminator_from_u(
                        img_usps, training=True) - 1) ** 2, axis=0)
                # on fake
                loss_discri_mnist_fake = tf.reduce_mean(
                    self.discriminator_from_m(
                        self.generator_from_u(
                            img_usps, training=True)) ** 2, axis=0)
                loss_discri_usps_fake = tf.reduce_mean(
                    self.discriminator_from_u(
                        self.generator_from_m(
                            img_mnist, training=True)) ** 2, axis=0)
                loss_discri_m = loss_discri_mnist_fake + loss_discri_mnist_real
                loss_discri_u = loss_discri_usps_fake + loss_discri_usps_real

            # get gradients
            discri_m_grads = tape_discri.gradient(
                loss_discri_m, self.discriminator_from_m.trainable_variables)
            discri_u_grads = tape_discri.gradient(
                loss_discri_u, self.discriminator_from_u.trainable_variables)

            # update weights
            self.discriminator_from_m_opti.apply_gradients(
                zip(discri_m_grads,
                    self.discriminator_from_m.trainable_variables)
            )
            self.discriminator_from_u_opti.apply_gradients(
                zip(discri_u_grads,
                    self.discriminator_from_u.trainable_variables)
            )

            if iter % frequence_sample == 0:
                print("loss_discri_m : ", loss_discri_m.numpy(),
                      " loss_discri_u : ", loss_discri_u.numpy())

            # ===============================
            # ====== update generators ======
            # ===============================
            with tf.GradientTape(persistent=True) as tape_gen:
                cycle_img_mnist = self.generator_from_u(
                    self.generator_from_m(img_mnist), training=True)
                cycle_img_usps = self.generator_from_m(
                    self.generator_from_u(img_usps), training=True)

                # cycle loss
                cycle_loss_usps = self.lambda_cycle * \
                    tf.reduce_mean(tf.abs(img_mnist - cycle_img_mnist))
                cycle_loss_mnist = self.lambda_cycle * \
                    tf.reduce_mean(tf.abs(img_usps - cycle_img_usps))

                # loss generator from mnist
                discri_u = self.discriminator_from_u(
                    self.generator_from_m(
                        img_mnist, training=True), training=True)
                loss_gen_mnist = tf.reduce_mean((discri_u - 1) ** 2, axis=0)
                loss_gen_mnist = loss_gen_mnist + cycle_loss_mnist

                # loss generator from usps
                discri_m = self.discriminator_from_m(
                    self.generator_from_u(
                        img_usps, training=True), training=True)
                loss_gen_usps = tf.reduce_mean((discri_m - 1) ** 2, axis=0)
                loss_gen_usps = loss_gen_usps + cycle_loss_usps

            # get gradients
            gen_m_grads = tape_gen.gradient(
                loss_gen_mnist, self.generator_from_m.trainable_variables)
            gen_u_grads = tape_gen.gradient(
                loss_gen_usps, self.generator_from_u.trainable_variables)

            # update weights
            self.generator_from_m_opti.apply_gradients(
                zip(gen_m_grads, self.generator_from_m.trainable_variables)
            )
            self.generator_from_u_opti.apply_gradients(
                zip(gen_u_grads, self.generator_from_u.trainable_variables)
            )

            if iter % frequence_sample == 0:
                print("loss_gen_mnist : ", loss_gen_mnist.numpy(),
                      " loss_gen_usps : ", loss_gen_usps.numpy())

            if iter % frequence_sample == 0:
                self.sample_images()

    def sample_images(self):

        # sample and generate images
        idx_mnist = np.random.randint(0, len(self.X_mnist), 4)
        img_mnist = (self.X_mnist[idx_mnist] + 1) * 127.5
        img_from_mnist = (self.generator_from_m(img_mnist) + 1) * 127.5
        idx_usps = np.random.randint(0, len(self.X_usps), 4)
        img_usps = (self.X_usps[idx_usps] + 1) * 127.5
        img_from_usps = (self.generator_from_u(img_usps) + 1) * 127.5

        # displat them
        fig, axs = plt.subplots(4, 4)

        for i in range(4):
            axs[i, 0].imshow(img_mnist[i, :, :, 0], cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(img_from_mnist[i, :, :, 0], cmap='gray')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(img_usps[i, :, :, 0], cmap='gray')
            axs[i, 2].axis('off')
            axs[i, 3].imshow(img_from_usps[i, :, :, 0], cmap='gray')
            axs[i, 3].axis('off')

        plt.show()
