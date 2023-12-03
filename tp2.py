import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.mixture import GaussianMixture


def loadImages(dir):
    """
    Load toutes les images jpeg d'un directory dans un array

    :param dir: path du directory.
    """
    images_array = []
    for filename in sorted(os.listdir(dir), reverse=True):
        img_path = os.path.join(dir, filename)
        if img_path.lower().endswith(".jpeg"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Réduire la résolution de moitié
            height, width = img.shape[:2]
            new_height = height // 2
            new_width = width // 2
            img_resized = cv2.resize(img, (new_width, new_height))

            images_array.append(img_resized)
            # images_array.append(img)
    return images_array


def plotImages(img_detection, img_librairie, output_path):
    """
    Affiche et sauvegarde une figure contenant les images de chacun des 2 arrays

    :param img_detection: array d'images de detection.
    :param img_libraire: array d'images de la librairie.
    :param output_path: path pour sauvegarder la figure.
    """

    num_arrays = 2  # Assuming you always have two arrays

    max_len = max(len(img_detection), len(img_librairie))

    fig, axes = plt.subplots(num_arrays, max_len, figsize=(12, 8))

    for i in range(num_arrays):
        for j in range(max_len):
            ax = axes[i, j]
            ax.axis("off")  # Turn off axis labels and ticks
            img_array = img_detection if i == 0 else img_librairie
            if j < len(img_array):
                ax.imshow(img_array[j], cmap="gray")

    fig.suptitle("Image Arrays", fontsize=16)
    plt.savefig(os.path.join(output_path, "images"))
    plt.show()


def plot_binary_images(binary_images, output_path):
    """
    Affiche et sauvegarde une figure contenant les images binaires

    :param binary_images: array d'images binaires.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(binary_images)
    rows = (num_images + 1) // 3  # Number of rows for subplots
    cols = 3  # Two columns per row

    plt.figure(figsize=(12, 8))

    for i, binary_image in enumerate(binary_images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(binary_image, cmap="gray")
        plt.title(f"Binary Image {i}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_trees(trees, output_path):
    """
    Affiche et sauvegarde une figure contenant les arbres de toutes les images

    :param trees: array d'arbres d'adjacence.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(trees)
    rows = (num_images + 1) // 3  # Number of rows for subplots
    cols = 3  # Three columns per row

    plt.figure(figsize=(15, 10))

    for i, tree in enumerate(trees, 1):
        # Apply your diffusion filling algorithm here
        G = nx.DiGraph()
        for node, parent in tree.items():
            G.add_node(node)
            if parent != -1:
                G.add_edge(parent, node)

        plt.subplot(rows, cols, i)
        pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos,
            with_labels=True,
            font_weight="bold",
            node_size=700,
            node_color="skyblue",
            font_color="black",
            arrowsize=20,
        )

        plt.title(f"Adjacency Tree for Image {i}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_trees.png"))
    plt.show()


def plot_tree(tree, output_path):
    G = nx.DiGraph()

    for region_id, region in tree.items():
        G.add_node(region_id, color=region.color)  # Ajouter un nœud pour chaque région

        if region.parent is not None:
            G.add_edge(
                region.parent, region_id
            )  # Ajouter une arête du parent à l'enfant

    pos = nx.spring_layout(G)  # Ajuster la disposition des nœuds

    node_colors = [region.color for region_id, region in tree.items()]
    nx.draw(G, pos, node_color=node_colors, with_labels=True)
    plt.show()


def gaussian_mixture_segmentation(image):
    """
    Sépare une image en deux classes en utilisant la méthode de mélange de gaussienne.

    :param image: Matrice représentant l'image.
    """

    n_classes = 2
    flattened_image = image.flatten().reshape(-1, 1)

    # Trouver les 2 gaussiennes de l'image
    gmm = GaussianMixture(
        n_components=n_classes,
        random_state=42,
    )
    gmm.fit(flattened_image)

    # Predire les classes des pixels de l'image selon la gaussienne
    labels = gmm.predict(flattened_image)
    segmented_image = labels.reshape(image.shape)

    return segmented_image


class PixelRegion:
    """
    Initialise un objet PixelRegion avec la couleur spécifiée et une référence optionnelle à un parent.

    :param color: Couleur du pixel dans la région.
    :param parent: Référence optionnelle à la région parente.
    """

    def __init__(self, color, parent=None):
        self.color = color
        self.parent = parent
        self.pixels = []
        self.children = []

    def add_pixel(self, pixel):
        self.pixels.append(pixel)


def flood_fill(
    image, start_x, start_y, current_color, region, visited_color, adjacency_tree
):
    """
    Remplit une région dans une image en utilisant la méthode de remplissage par pile.

    :param image: Matrice représentant l'image.
    :param start_x: Coordonnée x du point de départ.
    :param start_y: Coordonnée y du point de départ.
    :param current_color: Couleur du pixel d'origine à remplir.
    :param region: Identifiant de la région en cours de remplissage.
    :param border_color: Couleur qui marque la frontière de la région.
    :param adjacency_tree: Arbre d'adjacence pour enregistrer les relations entre les régions.
    """

    stack = [(start_x, start_y)]
    cpt = 0

    while stack:
        x, y = stack.pop()

        # Vérifier les limites de l'image
        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]:
            continue

        # Vérifier si le pixel est déjà visité
        if image[x, y] == visited_color:
            if region != 1:
                if adjacency_tree[region].parent == None:
                    # Mettre à jour la relation parent-enfant si le pixel est déjà visité
                    parent_region = region_at_pixel(x, y, adjacency_tree)
                    # adjacency_tree[parent_region].children.append(region)
                    adjacency_tree[region].parent = parent_region
            continue

        # Ignorer les pixels qui ne sont pas de la couleur actuelle
        if image[x, y] != current_color:
            continue

        # Marquer le pixel comme visité
        image[x, y] = np.int_(-1)

        # Ajouter le pixel
        cpt += 1
        if cpt % 100000 == 0:
            print(cpt)
        adjacency_tree[region].add_pixel([x, y])

        # explorer les 8 voisins
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # position du pixel courant
                    continue
                stack.append((x + dx, y + dy))  # positions des voisins


def build_adjacency_tree(image):
    """
    Construit un arbre d'adjacence à partir d'une image binaire.

    :param image: Matrice représentant l'image.
    """

    adjacency_tree = {}
    region_counter = 1

    print_frequency = 10
    pixel_cpt = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= 0:  # pixel non visité
                # initialisation de la nouvelle région
                adjacency_tree[region_counter] = PixelRegion(image[i, j])

                # remplissage de la région
                flood_fill(image, i, j, image[i, j], region_counter, -1, adjacency_tree)

                # fin de la région actuelle
                region_counter += 1

            # pixel_cpt += 1
            # if pixel_cpt % print_frequency == 0:
            #     print(
            #         f"Pixels traités jusqu'à ({i}, {j}). Région actuelle : {region_counter}"
            #     )

    return adjacency_tree


def region_at_pixel(x, y, adjacency_tree):
    """
    Retourne l'identifiant de la région à laquelle appartient le pixel aux coordonnées (x, y).

    :param x: Coordonnée x du pixel.
    :param y: Coordonnée y du pixel.
    :param adjacency_tree: Arbre d'adjacence contenant les régions.
    :return: Identifiant de la région à laquelle appartient le pixel.
    """
    print("ici")
    for region_id, pixel_region in adjacency_tree.items():
        if [x, y] in pixel_region.pixels:
            return region_id

    return None


def detect_markers(image):
    # Implementation for marker detection
    # Your code here
    pass


def remove_small_regions(img, min_size):
    img = img.astype(np.uint8)
    # Effectuer une ouverture (érosion suivie de dilatation)
    kernel = np.ones((3, 3), np.uint64)
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Trouver les composants connectés
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img_opened, connectivity=8)

    # Identifier les régions à conserver
    valid_regions = stats[:, 4] >= min_size

    # Créer une image résultante avec les régions valides
    img_result = np.zeros_like(img)
    for label in range(1, len(stats)):
        if valid_regions[label]:
            img_result[labels == label] = 1

    return img_result


def main():
    """
    Ce programme permet de détecter des marqueurs topologiques dans une série d'images à partir d'une librairie de marqueurs.
    """

    # Specify the directory where your images are located
    detection_dir = "detection/detection/"
    librairie_dir = "librairie2/librairie2/"

    # Specify the directory where to store images
    output_dir = "images_resultats/"
    os.makedirs(output_dir, exist_ok=True)

    # Load all image files in the directories
    print("----------------- LOADING IMAGES -----------------")

    img_detection = loadImages(detection_dir)
    print("Detections images are loaded")

    img_librairie = loadImages(librairie_dir)
    print("Librairie images are loaded")

    # plotImages(
    #     img_detection,
    #     img_librairie,
    #     output_dir,
    # )

    print("Loading done")

    # Segment the library images into two classes by mixing Gaussians
    print("----------------- SEGMENTATION OF LIBRARY IMAGES -----------------")

    # binary_images = []
    # for i, image in enumerate(img_librairie):
    #     binary_image = gaussian_mixture_segmentation(image)
    #     binary_images.append(binary_image)
    #     print(f"Segmentation of image {i} done")

    # plot_binary_images(
    #     binary_images, os.path.join(output_dir, "binary_images_combined.png")
    # )

    print(img_librairie[0].shape[0])
    print(img_librairie[0].shape[1])
    binary_image = gaussian_mixture_segmentation(img_librairie[1])
    processed_img = remove_small_regions(binary_image, 50)

    print(processed_img.shape[0])
    print(processed_img.shape[1])

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap="gray")
    plt.title("Image Originale")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap="gray")
    plt.title("Image Traitée")
    plt.axis("off")

    plt.show()

    # create the topological representation of the five library markers
    print("----------------- LIBRARY TOPOLOGY -----------------")

    # adjacency_trees = []
    # for i, image in enumerate(binary_images):
    #     adjacency_trees.append(build_adjacency_tree(image))
    #     print(f"Adjacency tree of image {i} done")

    # binary_image = np.array(
    #     [
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    #         [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    #         [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     ]
    # )

    adjacency_tree = build_adjacency_tree(binary_image)
    print(adjacency_tree)

    plot_tree(adjacency_tree, os.path.join(output_dir, "combined_trees.png"))

    # segment the detection images acquired by mixing Gaussians into two classes
    print("----------------- SEGMENTATION OF DETECTION IMAGES -----------------")
    # for i, image in enumerate(img_detection):
    #     binary_image = gaussian_mixture_segmentation(image)
    #     plot_binary_image(binary_image, os.path.join(output_dir, f"binary_image_{i}"))

    # create the topological representation of the five library markers
    print("----------------- COMPLETE LIBRARY IMAGE TOPOLOGY -----------------")
    # complete_library_topology = generate_adjacency_trees(
    #     img_librairie
    # )  # Assuming you choose the first image
    # plot_trees(complete_library_topology, output_dir)

    # detect the markers using topology trees
    print("----------------- DETECT MARKERS -----------------")
    # for i, image in enumerate(img_detection):
    #     markers_info = detect_markers(image)
    #     print(f"Markers info for detection image {i + 1}: {markers_info}")

    # Lorsqu'il est question de détection, on souhaite obtenir l'identifiant du marqueur (chiffre 1 à 5) de même
    # que la position et l'orientation de celui-ci sur l'image.
    # Pour l'orientation, vous pouvez considérer que l'image de la librairie correspond à une rotation nulle.


if __name__ == "__main__":
    main()