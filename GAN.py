from __future__ import absolute_import, division, print_function, unicode_literals #Not sure what this is for...

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # silence warnings

import tensorflow as tf

import numpy as np
import math
import os
from tensorflow import keras
from tensorflow.keras import layers
import time
from datetime import datetime

from IPython import display
import matplotlib.pyplot as plt

if 'base_dir' not in globals(): base_dir = '.' # define if doesn't exist (just for convenience so a prior colab cell can set it.)

class GAN:
    """ Represents a Generative Adversarial Neural Net with a generator and discriminator. """
    # class variables

    # This method returns a helper function to compute cross entropy loss
    _cross_entropy = keras.losses.BinaryCrossentropy(
        from_logits=True,
        # label_smoothing=0.2, # Bends labels from 0 and 1 towards 0.5. This is supposed to help regularize somehow.
    )
    # A object that sets the initial random weights for the model.
    # Recommended settings from https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
    _r_norm = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def __init__(self,
        noise_dim = 100, save_every=4, label_noise=0.05,
        log_file=f"{base_dir}/training_log.txt",
        checkpoint_dir=f"{base_dir}/checkpoints"
    ):
        self.noise_dim = noise_dim
        self.save_every = save_every
        self.label_noise = label_noise
        self.log_file = log_file

        self.generator = self._make_generator_model()
        # Recommended settings from https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
        self.generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.discriminator = self._make_discriminator_model()
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # For saving the model
        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator,
            generator_optimizer = self.generator_optimizer,
            discriminator = self.discriminator,
            discriminator_optimizer = self.discriminator_optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep = 5)

    def _make_generator_model(self):
        """
        Creates a generator model.
        The model will take in a noise vector, and output an image.
        """

        model = keras.Sequential()
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(self.noise_dim,), kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 8*8*256) # Note: None is the batch size

        model.add(layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256)

        model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 16, 16, 512)

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 32, 32, 256)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 64, 64, 128)

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 128, 128, 64)

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh', kernel_initializer=GAN._r_norm))
        assert model.output_shape == (None, 128, 128, 3)

        return model

    def _make_discriminator_model(self):
        """
        Makes a discriminator model.
        The model will take in an image, and return a logit of whether it thinks the image if fake (0) or real (1).
        Run through sigmoid to get the final probability.
        """
        model = keras.Sequential()

        model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', kernel_initializer=GAN._r_norm,
                                    input_shape=[128, 128, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))
        assert model.output_shape == (None, 128, 128, 64)

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))
        assert model.output_shape == (None, 64, 64, 128)

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))
        assert model.output_shape == (None, 32, 32, 256)

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))
        assert model.output_shape == (None, 16, 16, 512)

        model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same', kernel_initializer=GAN._r_norm))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))
        assert model.output_shape == (None, 8, 8, 1024)

        model.add(layers.Flatten())
        model.add(layers.Dense(1, kernel_initializer=GAN._r_norm))
        assert model.output_shape == (None, 1)

        return model

    @classmethod
    def discriminator_loss(cls, real_output, fake_output, noise):
        """
        Determines the loss for the discriminator model.
        real_output is the discriminators output on real images, and fake_output is the discriminators output for fake images
        """
        real_labels = noisy_labels(tf.constant(1, dtype=tf.float32), real_output.shape, noise)
        real_loss = cls._cross_entropy(real_labels, real_output) # compare real_output against all 1s + noise

        fake_labels = noisy_labels(tf.constant(0, dtype=tf.float32), fake_output.shape, noise)
        fake_loss = cls._cross_entropy(fake_labels, fake_output) # compare fake_output against all 0s + noise

        total_loss = real_loss + fake_loss
        return total_loss

    @classmethod
    def generator_loss(cls, fake_output):
        """
        Determines the generators loss.
        The generator wants the discriminator's to guess wrong on its counterfeits (ie. when the discriminator outputs 1).
        """
        return cls._cross_entropy(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    # see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md and
    # https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/ for limitations on this.
    @tf.function
    def train_step(self, images, dscr_correct_prcnt):
        """
        Preforms one training step on one batch. images is a tensor of real images.
        Will run the generator on noise, and the run the discriminator on the images given and the generator output,
        calculating loss of the generator based on how many of its fakes got past the discriminator.
        Returns a tensor of (gen_loss, disc_loss, disc_percentage_correct)
        """
        noise = tf.random.normal([images.shape[0], self.noise_dim])

        # throttle = dscr_correct_prcnt > 0.80 # Throttle discriminator, its getting too good.

        # if throttle:
        #     # gradient tape records results from a function and calculates derivates for it.
        #     with tf.GradientTape() as gen_tape:
        #         # Run the generator
        #         generated_images = self.generator(noise, training=True)

        #         # Let the discriminator try to flag fakes out of our real images and our generated_images
        #         # https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/ recommends not shuffling real and fake together.
        #         real_output = self.discriminator(images,           training=False)
        #         fake_output = self.discriminator(generated_images, training=False)

        #         # Calculate the loss. We have to do this manually instead of letting keras do it for us with .fit(), since we have to determine the loss
        #         # based of the discriminator's output.
        #         gen_loss  = GAN.generator_loss(fake_output)
        #         disc_loss = GAN.discriminator_loss(real_output, fake_output)

        #     # Calculate the gradients from the loss and apply them
        #     gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #     self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        # else:
        #     # gradient tape records results from a function and calculates derivates for it.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Run the generator
            generated_images = self.generator(noise, training=True)

            # Let the discriminator try to flag fakes out of our real images and our generated_images
            # https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/ recommends not shuffling real and fake together.
            real_output = self.discriminator(images,           training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate the loss. We have to do this manually instead of letting keras do it for us with .fit(), since we have to determine the loss
            # based of the discriminator's output.
            gen_loss  = GAN.generator_loss(fake_output)
            disc_loss = GAN.discriminator_loss(real_output, fake_output, self.label_noise)

        # Calculate the gradients from the loss and apply them
        gradients_of_generator     = gen_tape.gradient(gen_loss,   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply the gradients to the models.
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # log the actual percentage of fakes the discriminator marked correctly.
        correct_count = 0
        for output in fake_output:
            if output[0] < 0: # Marked as fake, since discriminator outputs a logit.
                correct_count += 1

        return tf.tuple([gen_loss, disc_loss, correct_count / fake_output.shape[0]])

    def train(self, dataset, epochs):
        """
        Trains the GAN for the given number of epochs, using the given dataset.
        Restores the model from a previous training session if there is one saved. Saves the model every few epochs.
        Also saves a few sample images for each epoch.
        """
        self.log("Training...")

        for epoch in range(epochs):
            start = time.time()

            loss_sums = [0, 0, 0] # gen_loss, disc_loss, disc_percentage_correct
            batches_in_epoch = 0
            # throttled_count = 0
            last_correct_percentage = tf.constant(0, dtype=tf.float64)

            for image_batch, label_batch in dataset:
                losses = self.train_step(image_batch, last_correct_percentage )
                # if (last_correct_percentage > 0.8):
                #     throttled_count += 1
                batches_in_epoch += 1
                last_correct_percentage = losses[2]
                for i in range(len(losses)): loss_sums[i] += losses[i]
            # Save the model and sample images every couple epochs
            if (epoch + 1) % self.save_every == 0:
                self.checkpoint_manager.save()
                self.save_sample_images(epoch + 1) # Produce sample_images

            self.log(f"Time for Epoch {epoch + 1}: {time.time() - start} sec")

            self.log(f"Average losses for Epoch {epoch + 1}: (generator: {loss_sums[0]/batches_in_epoch}, discriminator: {loss_sums[1]/batches_in_epoch}).")
            self.log(f"Discriminator Accuracy on fakes for Epoch {epoch + 1}: {loss_sums[2]/batches_in_epoch}.")
            # self.log(f"Discriminator throttled {throttled_count} out of {batches_in_epoch} batches.")

        # Save after the final epoch
        self.checkpoint_manager.save()

        self.log("DONE!")

    def restore(self):
        """ Tries to restore GAN state from checkpoint. Returns True if restor successful, false otherwise. """
        # expect_partial() silences the warnings if you don't use all the restored objects.
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
        if self.checkpoint_manager.latest_checkpoint:
            self.log(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            return True
        else:
            self.log("Initializing from scratch.")
            return False

    def log(self, message):
        """ Prints to console and logs to self.log_file. """
        if (self.log_file):
            log = open(self.log_file, 'a')
            log.write(message + "\n")
            print(message)
            log.close()

    def generate(self, count):
        """ Returns an array of generated images. """
        noise = tf.random.normal([count, self.noise_dim])
        images = self.generator(noise, training=False)
        images = images[:, :, :, :] * 127.5 + 127.5 # denormalize image.
        images = tf.dtypes.cast(images, tf.uint8)
        return images

    def save_sample_images(self, epoch, sample_count = 4):
        """
        Saves sample_count images generated by the model.
        Saves under name "generated_image_epoch{epoch}{a, b, c, etc.}.jpg
        """
        images = self.generate(sample_count)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for index, image in enumerate(images):
            image = tf.image.encode_jpeg(image)
            sub = "" if sample_count == 1 else chr(97 + index) if sample_count < 26 else f"-{index}"
            tf.io.write_file(f"{base_dir}/sample_images/generated_image_{now}_epoch{epoch}{sub}.jpg", image)

    def display_samples(self, sample_count = 16):
        images = self.generate(sample_count)

        display_width = math.ceil(math.sqrt(sample_count))

        fig = plt.figure(figsize=(display_width, display_width))
        for i in range(images.shape[0]):
            plt.subplot(display_width, display_width, i+1)
            plt.imshow(images[i, :, :, :])
            plt.axis('off')

        plt.show()


def noisy_labels(label, shape, noise):
    labels = tf.fill(shape, label)
    flip = lambda e: 1 - e if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) < noise else e
    return tf.map_fn(flip, labels, dtype=tf.float32)

def load_image(file_path):
    """
    Loads an image from a file path into a (tensor image, label) tuple.
    Normalizes pixels into range [-1.0, 1.0]. Image directory determines the class label.
    """
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.dtypes.cast(img, tf.float32)
    img = (img - 127.5) / 127.5 # normalize to [-1, 1] range.
    # img = tf.image.resize(img, [128, 128])

    parts = tf.strings.split(file_path, os.path.sep) # second to last is the directory, and will be used as class name.
    label = tf.strings.lower(parts[-2])
    return img, label

def load_data():
    """
    Returns image data as a tf.data.Dataset
    Pulls data from the given folder.
    """
    BATCH_SIZE = 128

    # Pull a list of file names matching a glob, in random order.
    image_datset = tf.data.Dataset.list_files(f"{base_dir}/Data/ProcessedImages/Giraffe/*")

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # num_parallel_calls=tf.data.experimental.AUTOTUNE is supposed to let it adjust dynamically. However, it seems
    # to eat all your memory and then crash.
    image_datset = image_datset.map(load_image, num_parallel_calls=3)

    image_datset = image_datset.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    image_datset = image_datset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return image_datset

if __name__ == "__main__":
    gan = GAN()
    gan.restore()
    gan.train(load_data(), 100)