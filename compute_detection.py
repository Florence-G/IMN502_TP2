import os
from utils.adjacency_trees import *
from utils.detection import *
from utils.utils import *
from utils.segmentation import *


def main():
    """
    Ce programme permet de détecter des marqueurs topologiques dans une série d'images à partir d'une librairie de marqueurs.
    """

    output_dir = "resultats/"
    os.makedirs(output_dir, exist_ok=True)

    print("----------------- DETECT MARKERS -----------------")
    print("----------------- LOAD TREES -----------------")

    markers_trees_dir = "librairy_trees/"

    markers_trees = load_trees(markers_trees_dir)
    print("Markers trees are loaded")

    detection_trees_dir = "detection_trees/"

    detection_trees = load_trees(detection_trees_dir)
    print("Detections trees are loaded")

    print("----------------- LOAD IMAGES -----------------")

    images_dir = "detection/detection"
    images = loadImages(images_dir)

    print("Images are loaded")

    print("----------------- COMPUTE DETECTION -----------------")

    for i, detection_tree in enumerate(detection_trees):
        if i >= 1:
            markers_infos = detect_markers_using_topology(markers_trees, detection_tree)

            visualize_markers(
                images[i],
                markers_infos,
                os.path.join(output_dir, f"detection_finale_{i}.png"),
            )
    print("----------------- DETECTION COMPLETED -----------------")


if __name__ == "__main__":
    main()
