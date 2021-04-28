import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Dense, Reshape, Input, Conv2D, Flatten,
                                     UpSampling2D, LayerNormalization)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class GLU:

    def __call__(self, x):
        ''' Assumes (batch_size, dim) or (batch_size, h, w, channels). '''
        a, b = tf.split(x, 2, axis=-1)
        a = a * tf.sigmoid(b)
        return a


class OTGAN:

    ''' OTGAN architecture as described in Salimans et al.

    Again, this class is specificaly designed for MNIST, but easily adaptable
    to more complicated settings. '''

    def __init__(self, latent_dim=100, n_gen=2, path_save="OTGAN"):
        self.path_save = path_save
        self.latent_dim = latent_dim
        self.n_gen = n_gen
        self.shape_img = (28, 28, 1)

        # build
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # optimizers
        self.generator_opti = Adam(learning_rate=3e-4, beta_1=0.5)
        self.discriminator_opti = Adam(learning_rate=3e-4, beta_1=0.5)

        # data
        self.X_train = self.load_data()

    def build_generator(self):
        ''' Architecture as described by the authors
        (but adapted to MINST). '''
        input = Input((self.latent_dim,))
        x = Dense(16384 * 2)(input)
        x = GLU()(x)
        x = Reshape((4, 4, 1024))(x)
        x = UpSampling2D()(x)
        x = Conv2D(1024, kernel_size=(2, 2))(x)
        x = GLU()(x)
        x = UpSampling2D()(x)
        x = Conv2D(512, kernel_size=(5, 5), padding="same")(x)
        x = GLU()(x)
        x = UpSampling2D()(x)
        x = Conv2D(256, kernel_size=(5, 5), padding="same")(x)
        x = GLU()(x)
        output = Conv2D(1, kernel_size=(5, 5), padding="same",
                        activation='tanh')(x)
        return Model(input, output)

    def build_discriminator(self):
        ''' Architecture as described by the authors
        (but adapted to MINST). '''
        input = Input(self.shape_img)
        x = Conv2D(128, (5, 5), padding='same')(input)
        x = tf.nn.crelu(x)
        x = Conv2D(256, (5, 5), strides=2, padding='same')(x)
        x = tf.nn.crelu(x)
        x = Conv2D(512, (5, 5), strides=2, padding='same')(x)
        x = tf.nn.crelu(x)
        x = Conv2D(1024, (3, 3), strides=2, padding='same')(x)
        x = tf.nn.crelu(x)
        x = Flatten()(x)
        output = LayerNormalization()(x)
        return Model(input, output)

    @staticmethod
    def load_data():
        ''' Utils function to loas MINST. '''
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train.astype(np.float32) / 127.5 - 1
        return tf.expand_dims(X_train, axis=3)

    @staticmethod
    def cosine_similarity(a, b):
        ''' Cosine similarity. '''
        norm_a = tf.norm(a, ord=2, axis=1, keepdims=True)
        norm_b = tf.norm(b, ord=2, axis=1, keepdims=True)
        ratio = tf.linalg.matmul(a, b, transpose_b=True) / \
            tf.linalg.matmul(norm_a, norm_b, transpose_b=True)
        return 1 - ratio

    @staticmethod
    def sinkhorn(a, b, M, reg=0.01, iters=100):
        ''' Sinkhorn algorithm. '''
        v = tf.ones_like(b)

        K = tf.exp(-M / reg)
        for _ in range(iters):
            u = a / (tf.matmul(K, v) + 1e-8)
            v = b / (tf.matmul(K, u, transpose_a=True) + 1e-8)

        return tf.stop_gradient(tf.matmul(tf.linalg.diag(u[:, 0]),
                                tf.matmul(K, tf.linalg.diag(v[:, 0]))))

    def train(self, epochs, batch_size=32):
        ''' Training loop described in the architecture. '''

        _len_dataset = len(self.X_train)
        loss_training = []
        for epoch in range(epochs):
            print(f"================ EPOCH {epoch} ================")

            for t in range(_len_dataset // batch_size):

                # sample real (X and X prime)
                idx_1 = tf.random.uniform([batch_size], 0, _len_dataset,
                                          tf.int32)
                idx_2 = tf.random.uniform([batch_size], 0, _len_dataset,
                                          tf.int32)

                X_1 = tf.gather(self.X_train, idx_1, axis=0)
                X_2 = tf.gather(self.X_train, idx_2, axis=0)

                # sample latent z
                z_1 = tf.random.normal((batch_size, self.latent_dim))
                z_2 = tf.random.normal((batch_size, self.latent_dim))
                with tf.GradientTape() as tape:

                    # build fake images (Y and Y prime)
                    Y_1, Y_2 = self.generator(z_1), self.generator(z_2)

                    # calculate discriminator on fake and real
                    discri_X_1 = self.discriminator(X_1)
                    discri_X_2 = self.discriminator(X_2)
                    discri_Y_1 = self.discriminator(Y_1)
                    discri_Y_2 = self.discriminator(Y_2)

                    # calculate cost matrices
                    cost_X_1_X_2 = self.cosine_similarity(discri_X_1,
                                                          discri_X_2)
                    cost_X_1_Y_1 = self.cosine_similarity(discri_X_1,
                                                          discri_Y_1)
                    cost_X_1_Y_2 = self.cosine_similarity(discri_X_1,
                                                          discri_Y_2)
                    cost_X_2_Y_1 = self.cosine_similarity(discri_X_2,
                                                          discri_Y_1)
                    cost_X_2_Y_2 = self.cosine_similarity(discri_X_2,
                                                          discri_Y_2)
                    cost_Y_1_Y_2 = self.cosine_similarity(discri_Y_1,
                                                          discri_Y_2)

                    # calculate optimal plans
                    a = tf.ones([batch_size, 1]) / batch_size

                    plan_X_1_X_2 = self.sinkhorn(a, a, cost_X_1_X_2)
                    plan_X_1_Y_1 = self.sinkhorn(a, a, cost_X_1_Y_1)
                    plan_X_1_Y_2 = self.sinkhorn(a, a, cost_X_1_Y_2)
                    plan_X_2_Y_1 = self.sinkhorn(a, a, cost_X_2_Y_1)
                    plan_X_2_Y_2 = self.sinkhorn(a, a, cost_X_2_Y_2)
                    plan_Y_1_Y_2 = self.sinkhorn(a, a, cost_Y_1_Y_2)

                    # calculate losses
                    loss_X_1_X_2 = tf.math.reduce_sum(plan_X_1_X_2 *
                                                      cost_X_1_X_2)
                    loss_X_1_Y_1 = tf.math.reduce_sum(plan_X_1_Y_1 *
                                                      cost_X_1_Y_1)
                    loss_X_1_Y_2 = tf.math.reduce_sum(plan_X_1_Y_2 *
                                                      cost_X_1_Y_2)
                    loss_X_2_Y_1 = tf.math.reduce_sum(plan_X_2_Y_1 *
                                                      cost_X_2_Y_1)
                    loss_X_2_Y_2 = tf.math.reduce_sum(plan_X_2_Y_2 *
                                                      cost_X_2_Y_2)
                    loss_Y_1_Y_2 = tf.math.reduce_sum(plan_Y_1_Y_2 *
                                                      cost_Y_1_Y_2)

                    loss = loss_X_1_Y_1 + loss_X_1_Y_2 + loss_X_2_Y_1 + \
                        loss_X_2_Y_2 - 2 * loss_X_1_X_2 - 2 * loss_Y_1_Y_2

                # get gradients
                if t % (self.n_gen + 1) == 0:
                    # notice the -1
                    discri_grads = -1 * tape.gradient(
                        loss, self.discriminator.trainable_variables)
                    self.discriminator_opti.apply_gradients(
                        zip(discri_grads,
                            self.discriminator.trainable_variables)
                    )

                else:
                    gen_grads = tape.gradient(
                        loss, self.generator.trainable_variables)
                    self.generator_opti.apply_gradients(
                        zip(gen_grads,
                            self.generator.trainable_variables)
                    )

                loss_training.append(loss.numpy().item())
            self.sample_images(epoch)
        self.save_model()
        return loss_training

    def save_model(self):
        self.generator.save_weights(self.path_save + "/generator/model")
        self.discriminator.save_weights(self.path_save +
                                        "/discriminator/model")

    @classmethod
    def from_pre_trained(cls, path_discriminator, path_generator,
                         latent_dim=100, n_gen=2):
        ''' Allows to build a trained model. '''
        tmp = cls(latent_dim=100, n_gen=2)
        tmp.generator.load_weights(path_generator)
        tmp.discriminator.load_weights(path_discriminator)
        return tmp

    def sample_images(self, epoch=None):
        ''' Same function as for DCGAN. '''
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
