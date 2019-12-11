from GAN import GAN
import tensorflow as tf
import os

# Generates samples from all checkpoints for celebA64v2 in the checkpoints folders, using the same seed.

noise = tf.random.normal([18, 100]) # We reuse the noise vector.


for folder in os.listdir("checkpoints"):
    if folder.startswith("celebA64v2"):
        gan = GAN(
            log_file=None,
            checkpoint_dir = f"checkpoints/{folder}",
        )
        for checkpoint in gan.checkpoint_manager.checkpoints:
            epoch = checkpoint.split("/")[-1]
            print(epoch)

            gan.checkpoint.restore(checkpoint).expect_partial()

            images = gan.generator(noise, training=False)
            images = images[:, :, :, :] * 127.5 + 127.5 # denormalize image.
            images = tf.dtypes.cast(images, tf.uint8)

            for index, image in enumerate(images):
                image = tf.image.encode_jpeg(image)
                tf.io.write_file(f"sample_images/celebA64-v2-sameSeed/generated_image_{epoch}_{index}.jpg", image)