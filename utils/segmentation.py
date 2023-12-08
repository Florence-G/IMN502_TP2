import numpy as np
import cv2
from sklearn.mixture import GaussianMixture


def gaussian_mixture_segmentation(image, random=70):
    """
    Sépare une image en deux classes en utilisant la méthode de mélange de gaussienne.

    :param image: Matrice représentant l'image.
    """

    n_classes = 2
    flattened_image = image.flatten().reshape(-1, 1)

    # Trouver les 2 gaussiennes de l'image
    gmm = GaussianMixture(
        n_components=n_classes,
        random_state=random,
    )
    gmm.fit(flattened_image)

    # Predire les classes des pixels de l'image selon la gaussienne
    labels = gmm.predict(flattened_image)
    segmented_image = labels.reshape(image.shape)

    return segmented_image


def remove_small_regions(img, min_size, max_aspect_ratio=2.0):
    """
    Retourne une image dont on a enlevé les zones isolées.

    :param image: Matrice représentant l'image.
    :param min_size: threshold de la grandeur des zones à conserver
    :param max_aspect_ratio: maximum aspect ratio allowed for regions
    """

    img = img.astype(np.uint8)

    # Effectuer une ouverture (érosion suivie de dilatation)
    kernel = np.ones((3, 3), np.uint64)
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Trouver les composants connectés
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img_opened, connectivity=8)

    # Identifier les régions à conserver en fonction de la taille et de l'aspect ratio
    valid_regions = np.logical_and(
        stats[:, 4] >= min_size, stats[:, 2] / stats[:, 3] <= max_aspect_ratio
    )

    # Créer une image résultante avec les régions valides
    img_result = np.zeros_like(img)
    for label in range(1, len(stats)):
        if valid_regions[label]:
            img_result[labels == label] = 1

    img_result = img_result.astype(np.int64)

    return img_result
