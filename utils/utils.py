import os
import cv2
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


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
    return images_array


def plotImages(images, output_path):
    """
    Affiche et sauvegarde une figure contenant les images des arrays

    :param images: array contenant des images.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(images)

    if num_images > 3:
        rows = (num_images + 1) // 3  # Number of rows for subplots
        cols = 3  # Two columns per row
    else:
        rows = 1
        cols = num_images

    plt.figure(figsize=(12, 8))

    for i, image in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap="gray")
        plt.title(f"Image {i}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_binary_images(binary_images, output_path):
    """
    Affiche et sauvegarde une figure contenant les images binaires

    :param binary_images: array d'images binaires.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(binary_images)

    if num_images > 3:
        rows = (num_images + 1) // 3  # Number of rows for subplots
        cols = 3  # Two columns per row
    else:
        rows = 1
        cols = num_images

    plt.figure(figsize=(12, 8))

    for i, binary_image in enumerate(binary_images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(binary_image, cmap="gray", interpolation="nearest")
        plt.colorbar(label="Valeurs")
        plt.title(f"Binary Image {i}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_region_images(region_images, output_path):
    """
    Affiche et sauvegarde une figure contenant les images

    :param region_images: array d'images.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(region_images)

    if num_images > 3:
        rows = (num_images + 1) // 3  # Number of rows for subplots
        cols = 3  # Two columns per row
    else:
        rows = 1
        cols = num_images

    plt.figure(figsize=(12, 8))

    for i, region_image in enumerate(region_images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(region_image, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Valeurs")
        plt.title(f"Region Image {i}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def save_binary_image(binary_image, output_path):
    """
    sauvegarde une image contenant les images binaires

    :param binary_image: image binaire (0 et 255).
    :param output_path: path pour sauvegarder l'image.
    """
    binary_image[binary_image == 1] = 255

    binary_image = binary_image.astype("uint8")

    cv2.imwrite(output_path, binary_image)


def load_binary_images(dir):
    """
    load les images binaires d'un dossier

    :param dir: dossier contenant les images
    """
    binary_images = []
    for filename in os.listdir(dir):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            # Utilisez cv2.imread pour charger l'image en niveau de gris (0 et 255)
            image_path = os.path.join(dir, filename)
            binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Assurez-vous que l'image est binaire (0 et 1)
            binary_image[binary_image > 0] = 1

            # Ajoutez l'image à la liste
            binary_images.append(binary_image)

    return binary_images


def plot_trees(trees, output_path):
    """
    Affiche et sauvegarde une figure contenant les arbres de toutes les images

    :param trees: array d'arbres d'adjacence.
    :param output_path: path pour sauvegarder la figure.
    """

    num_images = len(trees)

    if num_images > 3:
        rows = (num_images + 1) // 3  # Number of rows for subplots
        cols = 3  # Two columns per row
    else:
        rows = 1
        cols = num_images

    plt.figure(figsize=(15, 10))

    for i, tree in enumerate(trees, 1):
        plt.subplot(rows, cols, i)

        G = nx.DiGraph()

        for region_id, region in tree.items():
            G.add_node(region_id)  # Ajouter un nœud pour chaque région

            if region.parent is not None:
                G.add_edge(
                    region.parent, region_id
                )  # Ajouter une arête du parent à l'enfant

        pos = graphviz_layout(G, prog="dot")

        node_colors = [
            "black" if region.color == 0 else "white"
            for region_id, region in tree.items()
        ]

        nx.draw(
            G,
            pos,
            node_color=node_colors,
            with_labels=False,
            edgecolors="black",
            linewidths=2,
        )

        labels = {region_id: region_id for region_id in tree.keys()}

        nx.draw_networkx_labels(G, pos, labels=labels, font_color="aqua")

        plt.title(f"Adjacency Tree for Image {i}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path))
    plt.show()
