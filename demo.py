from GAN import *

print("Loading GAN...")
gan = GAN(
    log_file=None,
    checkpoint_dir = f"checkpoints/celebA64v2_epoch90",
)
gan.restore()
print("Displaying samples...")


image = gan.generate(1)[0]
fig = plt.imshow(image)

while plt.get_fignums():
    image = gan.generate(1)[0]
    fig.set_data(image)
    plt.pause(2)

