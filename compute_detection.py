import os
from utils.adjacency_trees import *
from utils.detection import *
from utils.utils import *
from utils.segmentation import *


def main():
    """
    Ce programme permet de détecter des marqueurs topologiques dans une série d'images à partir d'une librairie de marqueurs.
    """

    # Specify the directory where to store images
    output_dir = "resultats/"
    os.makedirs(output_dir, exist_ok=True)

    print("----------------- DETECT MARKERS -----------------")
    print("----------------- LOAD TREES -----------------")

    markers_trees_dir = "librairy_trees/"

    markers_trees = load_trees(markers_trees_dir)
    print("Markers trees are loaded")

    # plot_trees(markers_trees, "test.png")

    detection_trees_dir = "detection_trees/"

    detection_trees = load_trees(detection_trees_dir)
    print("Detections trees are loaded")

    # plot_trees(detection_trees, "test2.png")

    print("----------------- LOAD BINARY IMAGES -----------------")

    binary_images_dir = "binary_images/"
    binary_images = load_binary_images(binary_images_dir)
    # plot_binary_images(binary_images, "")

    print("Binary images are loaded")

    print("----------------- COMPUTE DETECTION -----------------")

    markers_infos = detect_markers_using_topology(markers_trees, detection_trees)

    print(markers_infos)

    visualize_markers(binary_images[7], markers_infos)

    # print(f"Markers info for detection image : {markers_info}")


if __name__ == "__main__":
    main()
