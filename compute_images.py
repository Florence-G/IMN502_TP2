import os
from utils.adjacency_trees import *
from utils.detection import *
from utils.utils import *
from utils.segmentation import *


def main():
    """
    Ce programme permet de générer les arbres d'adjacence des images de détection
    """

    output_dir = "resultats/"
    os.makedirs(output_dir, exist_ok=True)

    print("----------------- GENERATE DETECTION TREES -----------------")
    print("----------------- LOADING IMAGES -----------------")

    detection_dir = "detection/detection/"

    img_detection = loadImages(detection_dir)
    print("Detections images are loaded")

    plotImages(img_detection, os.path.join(output_dir, "images_detection.png"))

    print("----------------- SEGMENTATION OF DETECTION IMAGES -----------------")
    save_dir = "binary_images/"
    os.makedirs(save_dir, exist_ok=True)

    detection_binary_images = []
    random = [40, 40, 40, 40, 20, 20, 40, 40]
    for i, image in enumerate(img_detection):
        binary_image = gaussian_mixture_segmentation(image, random=random[i])
        processed_img = remove_small_regions(binary_image, 200)
        detection_binary_images.append(processed_img)
        save_binary_image(processed_img, os.path.join(save_dir, f"binary_image{i}.png"))
        print(f"Segmentation of image {i} done")

    plot_binary_images(
        detection_binary_images, os.path.join(output_dir, "detection_binary_images.png")
    )

    print("----------------- DETECTION TOPOLOGY -----------------")
    detection_trees_dir = "detection_trees/"
    os.makedirs(detection_trees_dir, exist_ok=True)

    detection_adjacency_trees = []
    detection_regions_images = []

    for i, image in enumerate(detection_binary_images):
        adjacency_tree, image_region = build_adjacency_tree(image)
        detection_adjacency_trees.append(adjacency_tree)
        detection_regions_images.append(image_region)
        save_adjacency_tree_to_json(
            adjacency_tree,
            os.path.join(detection_trees_dir, f"adjacency_tree{i}.json"),
        )
        print(f"Adjacency tree of image {i} done")

    plot_region_images(
        detection_regions_images, os.path.join(output_dir, "detection_regions.png")
    )
    plot_trees(
        detection_adjacency_trees, os.path.join(output_dir, "detection_trees.png")
    )

    print("----------------- GENERATE DETECTION TREES DONE -----------------")


if __name__ == "__main__":
    main()
