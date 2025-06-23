from PIL import Image
img = Image.open("/home/marta/Projects/tb/data/images/mch/1159/2/2023/01/01/1159_2_2023-01-01_1010.jpeg")  # Load an image
cropped_img = img.crop((0, 0, 512, 200))  # (left, upper, right, lower)
cropped_img.save("analysis/augmented/augmented_image.jpeg")  # Save the augmented image
img.save("analysis/augmented/original_image.jpeg")  # Save the original image