import os
import shutil


def reorganize_images_across_directories(source_dir, output_dir):
    """
    Reorganizes images from 'processed_tiles' subdirectories within the source directory.
    Creates a new directory structure in the output directory where images are grouped by name.

    Parameters:
    - source_dir: Root directory containing subdirectories with 'processed_tiles'.
    - output_dir: Target directory where reorganized images will be saved.

    Returns:
    - True if the operation is successful, False otherwise.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Dictionary to group images by name
        grouped_images = {}

        # Traverse the source directory
        for root, dirs, _ in os.walk(source_dir):
            for subdir in dirs:
                processed_tiles_dir = os.path.join(root, subdir, 'processed_tiles')

                if os.path.exists(processed_tiles_dir):
                    # Collect all images in the 'processed_tiles' directory
                    for file in os.listdir(processed_tiles_dir):
                        if file.endswith('.png'):
                            image_name = os.path.splitext(file)[0]
                            image_path = os.path.join(processed_tiles_dir, file)

                            if image_name not in grouped_images:
                                grouped_images[image_name] = []

                            grouped_images[image_name].append(image_path)

        # Organize images into the output directory
        for image_name, paths in grouped_images.items():
            # Create a directory for each image name
            image_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_dir, exist_ok=True)

            # Copy and rename images to the new directory
            for idx, src_path in enumerate(sorted(paths)):
                dest_path = os.path.join(image_dir, f"{idx}.png")
                shutil.copy2(src_path, dest_path)

        print(f"Images successfully reorganized into '{output_dir}'")
        return True

    except Exception as e:
        print(f"Error reorganizing images: {e}")
        return False



if __name__ == "__main__":

    source_dir = "outputs"
    reorganize_images_across_directories(source_dir=source_dir, output_dir="reorganized_outputs")