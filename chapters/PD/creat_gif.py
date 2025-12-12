# import imageio
import os
import pathlib
import imageio.v2 as imageio


# Path to the folder containing images
image_folder = 'ffffffff//samples//'
image_folder = 'ffffffff//samples//'
# List of image file names
image_files = ['image1.png', 'image2.png', 'image3.png'] 

image_files = list(pathlib.Path(image_folder).rglob(f"*.*png"))[0:10]

# Create a list to hold the images
images = []
for filename in image_files:
    #filepath = os.path.join(image_folder, filename)
    images.append(imageio.imread(filename))

# Save as gif
imageio.mimsave('output.gif', images, duration=0.7)  # duration is the time between frames in second