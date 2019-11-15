from __future__ import absolute_import, division, print_function, unicode_literals #Not sure what this is for...

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # silence warnings

import tensorflow as tf

import numpy as np
import os
from tensorflow.keras import layers
import time
from datetime import datetime

from IPython import display

if 'base_dir' not in globals(): base_dir = '.' # define if doesn't exist (just for convenience so a prior colab cell can set it.)

class GAN:
    """ Represents a Generative Adversarial Neural Net with a generator and discriminator. """
    # class variables

    # This method returns a helper function to compute cross entropy loss
    _cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, noise_dim = 100):
        self.noise_dim = noise_dim

        self.generator = self._make_generator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.discriminator = self._make_discriminator_model()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # For saving the model
        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator,
            generator_optimizer = self.generator_optimizer,
            discriminator = self.discriminator,
            discriminator_optimizer = self.discriminator_optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, f"{base_dir}/checkpoints", max_to_keep = 5)

    def _make_generator_model(self):
        """
        Creates a generator model.
        The model will take in a noise vector, and output an image.
        """

        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 512)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 128, 128, 3)

        return model

    def _make_discriminator_model(self):
        """
        Makes a discriminator model.
        The model will take in an image, and return whether it thinks the image if fake (0) or real (1).
        """
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=[128, 128, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        assert model.output_shape == (None, 64, 64, 1024)

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        assert model.output_shape == (None, 32, 32, 512)

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        assert model.output_shape == (None, 16, 16, 256)

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        assert model.output_shape == (None, 8, 8, 128)

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        assert model.output_shape == (None, 1)

        return model

    @classmethod
    def discriminator_loss(cls, real_output, fake_output):
        """
        Determines the loss for the discriminator model.
        real_output is the discriminators output on real images, and fake_output is the discriminators output for fake images
        """
        real_loss = cls._cross_entropy(tf.ones_like(real_output), real_output) # compare real_output against all 1s
        fake_loss = cls._cross_entropy(tf.zeros_like(fake_output), fake_output)# compare fake_output against all 0s
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
    def train_step(self, images):
        """
        Preforms one training step on one batch. images is a tensor of real images.
        Will run the generator on noise, and the run the discriminator on the images given and the generator output,
        calculating loss of the generator based on how many of its fakes got past the discriminator.
        """
        noise = tf.random.normal([images.shape[0], self.noise_dim])

        # gradient tape records results from a function and calculates derivates for it.
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
            disc_loss = GAN.discriminator_loss(real_output, fake_output)

        # Calculate the gradients from the loss.
        gradients_of_generator     = gen_tape.gradient(gen_loss,   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply the gradients to the models.
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
        """
        Trains the GAN for the given number of epochs, using the given dataset.
        Restores the model from a previous training session if there is one saved. Saves the model every few epochs.
        Also saves a few sample images for each epoch.
        """

        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        print("Training...")

        for epoch in range(epochs):
            start = time.time()

            for image_batch, label_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.save_sample_images(epoch + 1)

            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.checkpoint_manager.save()

            print (f"Time for Epoch {epoch + 1} is {time.time() - start} sec")

        # Save after the final epoch
        self.checkpoint_manager.save()
        print("DONE!")

    def save_sample_images(self, epoch, sample_count = 4):
        """
        Saves sample_count images generated by the model.
        Saves under name "generated_image_epoch{epoch}{a, b, c, etc.}.jpg
        """
        noise = tf.random.normal([sample_count, self.noise_dim])
        predictions = self.generator(noise, training=False)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for index, image in enumerate(predictions):
            image = predictions[index, :, :, :] * 127.5 + 127.5 # denormalize image.
            image = tf.dtypes.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image)
            sub = "" if sample_count == 1 else chr(97 + index) if sample_count < 26 else f"-{index}"
            tf.io.write_file(f"{base_dir}/sample_images/generated_image_{now}_epoch{epoch}{sub}.jpg", image)


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
    import matplotlib.pyplot as plt

    BATCH_SIZE = 1

    # Pull a list of file names matching a glob, in random order.
    image_datset = tf.data.Dataset.list_files(f"{base_dir}/Data/ProcessedImages/Giraffe/*")

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # num_parallel_calls=tf.data.experimental.AUTOTUNE is supposed to let it adjust dynamically. However, it seems
    # to eat all your memory and then crash.
    image_datset = image_datset.map(load_image, num_parallel_calls=2)

    image_datset = image_datset.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    image_datset = image_datset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return image_datset

gan = GAN()
gan.train(load_data(), 50)