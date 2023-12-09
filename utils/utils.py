import os
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np


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


def get_neighbors(position, size=10):
    """
    Renvoie les positions des voisins dans un carré de taille spécifiée autour de la position donnée.

    :param position: Tuple contenant les coordonnées (x, y) de la position.
    :param size: Taille du carré autour de la position.
    :return: Liste de tuples contenant les positions des voisins.
    """
    x, y = position
    neighbors = []

    for i in range(-size // 2, size // 2 + 1):
        for j in range(-size // 2, size // 2 + 1):
            neighbors.append((x + i, y + j))

    return neighbors


def visualize_markers(image, marker_data, output_path):
    """
    Visualise les marqueurs avec leurs positions et couleurs associées.

    :param marker_data: Liste de dictionnaires contenant les données des marqueurs.
    """

    color_mapping = {
        0: [1, 0, 0],  # Rouge
        1: [0, 1, 0],  # Vert
        2: [0, 0, 1],  # Bleu
        3: [0, 1, 1],  # Cyan
        4: [1, 0, 1],  # Magenta
    }

    image[image == 1] = 255

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    plt.figure(figsize=(12, 8))

    for marker in marker_data:
        marker_id = marker["marker_id"]
        position = tuple(map(int, marker["position"]))
        color = color_mapping.get(marker_id, [1, 1, 1])

        image_rgb[position] = np.array(color) * 255

        for neighbor_position in get_neighbors(position, 30):
            x, y = map(int, neighbor_position)
            image_rgb[x, y] = np.array(color) * 255

        # Ajouter une flèche au graphique
        orientation = marker.get("orientation", 0)

        # Tracer la flèche
        plt.arrow(
            position[1],
            position[0],
            orientation[1] - position[1],
            orientation[0] - position[0],
            color=color,
            head_width=30,
            head_length=30,
        )

    # Afficher l'image
    plt.imshow(image_rgb / 255.0)  # Scale to [0, 1] range
    plt.title("Visualisation des marqueurs")

    # Créer une légende
    legend_labels = [f"Marker {marker_id}" for marker_id in color_mapping.keys()]
    legend_colors = list(color_mapping.values())

    plt.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
            )
            for color in legend_colors
        ],
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    plt.savefig(output_path)
    plt.show()
