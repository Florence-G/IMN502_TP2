import os
from utils.adjacency_trees import *
from utils.detection import *
from utils.utils import *
from utils.segmentation import *


def main():
    """
    Ce programme permet de générer les arbres d'adjacence des marqueurs topologiques
    """

    output_dir = "resultats/"
    os.makedirs(output_dir, exist_ok=True)

    print("----------------- GENERATE MARKERS TREES -----------------")
    print("----------------- LOADING IMAGES -----------------")

    librairie_dir = "librairie2/librairie2/"

    img_librairie = loadImages(librairie_dir)
    print("Librairie images are loaded")

    plotImages(img_librairie, os.path.join(output_dir, "images_librairy.png"))

    print("----------------- SEGMENTATION OF LIBRARY IMAGES -----------------")

    binary_images = []
    for i, image in enumerate(img_librairie):
        binary_image = gaussian_mixture_segmentation(image)
        processed_img = remove_small_regions(binary_image, 1000)
        binary_images.append(processed_img)
        print(f"Segmentation of image {i} done")

    plot_binary_images(
        binary_images, os.path.join(output_dir, "markers_binary_images.png")
    )

    print("----------------- LIBRARY TOPOLOGY -----------------")
    libraire_trees_dir = "librairy_trees/"
    os.makedirs(libraire_trees_dir, exist_ok=True)

    adjacency_trees = []
    regions_images = []

    for i, image in enumerate(binary_images):
        adjacency_tree, image_region = build_adjacency_tree(image, 200)
        adjacency_trees.append(adjacency_tree)
        regions_images.append(image_region)
        save_adjacency_tree_to_json(
            adjacency_tree, os.path.join(libraire_trees_dir, f"adjacency_tree{i}.json")
        )
        print(f"Adjacency tree of image {i} done")

    plot_region_images(regions_images, os.path.join(output_dir, "markers_regions.png"))
    plot_trees(adjacency_trees, os.path.join(output_dir, "markers_trees.png"))

    print("----------------- GENERATE MARKERS TREES DONE -----------------")


if __name__ == "__main__":
    main()
