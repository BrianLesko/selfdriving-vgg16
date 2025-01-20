import os
from PIL import Image
import torchvision.transforms.functional as F

def augment_images(source_dir, augmented_left_dir, augmented_right_dir):
    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png"}

    # Create target directories if they don't exist
    os.makedirs(augmented_left_dir, exist_ok=True)
    os.makedirs(augmented_right_dir, exist_ok=True)

    # Helper function to check valid image files
    def is_image_file(filename):
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    # Process the "left" folder
    left_folder = os.path.join(source_dir, "left")
    for img_name in filter(is_image_file, os.listdir(left_folder)):
        img_path = os.path.join(left_folder, img_name)
        with Image.open(img_path) as img:
            flipped_img = F.hflip(img)  # Horizontally flip the image
            flipped_img.save(os.path.join(augmented_left_dir, f"flipped_{img_name}"))

    # Process the "right" folder
    right_folder = os.path.join(source_dir, "right")
    for img_name in filter(is_image_file, os.listdir(right_folder)):
        img_path = os.path.join(right_folder, img_name)
        with Image.open(img_path) as img:
            flipped_img = F.hflip(img)  # Horizontally flip the image
            flipped_img.save(os.path.join(augmented_right_dir, f"flipped_{img_name}"))

    print(f"Augmented images saved in {augmented_left_dir} and {augmented_right_dir}.")