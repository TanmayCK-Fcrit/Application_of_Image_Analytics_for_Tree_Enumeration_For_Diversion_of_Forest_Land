import os
import random
import cv2
import numpy as np

def get_random_image(image_dir):
    """Fetch a random image path from the given directory."""
    images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    random_image = random.choice(images)
    return os.path.join(image_dir, random_image)

def create_collage(coconut_dir, mango_dir, neem_dir, sandalwood_dir, output_dir, num_collages=100):
    """Generates a collage with images from the four categories."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_collages):
        try:
            # Load random images
            img1 = cv2.imread(get_random_image(coconut_dir))
            img2 = cv2.imread(get_random_image(mango_dir))
            img3 = cv2.imread(get_random_image(neem_dir))
            img4 = cv2.imread(get_random_image(sandalwood_dir))

            # Resize images to a fixed size (e.g., 256x256)
            size = (256, 256)
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)
            img3 = cv2.resize(img3, size)
            img4 = cv2.resize(img4, size)

            # Arrange in a 2x2 collage
            top_row = np.hstack((img1, img2))
            bottom_row = np.hstack((img3, img4))
            collage = np.vstack((top_row, bottom_row))

            # Save the collage
            collage_path = os.path.join(output_dir, f"collage_{i+1}.jpg")
            cv2.imwrite(collage_path, collage)
            print(f"Saved: {collage_path}")

        except Exception as e:
            print(f"Error creating collage {i+1}: {e}")

if __name__ == "__main__":
    # Directories containing tree images
    coconut_dir = "Common_coconut"
    mango_dir = "Common_mango"
    neem_dir = "Common_neem"
    sandalwood_dir = "Economical_sandalwood"
    
    # Output directory for dataset
    output_dir = "test_case"
    
    # Generate collages
    create_collage(coconut_dir, mango_dir, neem_dir, sandalwood_dir, output_dir, num_collages=100)
