import os

source_dir = ""
save_dir = ""

resize_size = 64

for dirs in [save_dir, os.path.join(save_dir, "CelebA")]:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

impath_list = [os.path.join(source_dir, image) for image in os.listdir(source_dir) if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png")]

