from GAN import GAN
import matplotlib.pyplot as plt

# settings
interpolation = 'none' # catrom
filterImages  = True
truncate = True

print("Loading GAN...")
gan = GAN(
    log_file=None,
    checkpoint_dir = f"checkpoints/celebA64v2_epoch124",
)
gan.restore()
print("Displaying samples...")

fig = plt.imshow([[[0, 0, 0]]], interpolation=interpolation)
plt.xticks([])
plt.yticks([])
plt.grid(False)

while plt.get_fignums():
    output = gan.generateAndEvaluate(10, truncate=truncate)

    i = 0
    while plt.get_fignums() and i < len(output):
        image, fooled = output[i]

        if (not filterImages) or fooled:
            fig.set_data(image)

            if fooled:
                plt.xlabel("Fooled Discriminator", fontdict={'color': 'green'})
            else:
                plt.xlabel("Did not fool Discriminator", fontdict={'color': 'red'})


            plt.pause(2)

        i += 1