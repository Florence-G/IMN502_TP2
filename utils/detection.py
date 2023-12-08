import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import cv2
import itertools


def detect_markers_using_topology(markers_trees, detection_tree):
    """
    Détecter la présence de marqueur grâce à leur arbre d'adjacence

    on souhaite obtenir l'identifiant du marqueur (chiffre 1 à 5) de même que la
    position et l'orientation de celui-ci sur l'image. Pour l'orientation, vous
    pouvez considérer que l'image de la librairie correspond à une rotation nulle.

    :param adjacency_tree: The adjacency tree of a segmented image.
    :return: List of markers with their properties.
    """

    markers = []

    for i, markers_tree in enumerate(markers_trees):
        # Notez l'utilisation de enumerate pour obtenir l'indice i et l'élément markers_tree
        mapping = is_subtree(detection_tree[7], markers_tree)
        print(mapping)

        # Vérifiez si la correspondance est trouvée avant d'accéder à la deuxième clé
        if mapping is not None and len(mapping) > 1:
            keys_list = list(mapping.keys())
            second_element = mapping[keys_list[1]]
            position = detection_tree[0][second_element].get_barycentre()

            markers.append(
                {
                    "marker_id": i,
                    "position": position,
                    "orientation": 0,  # Ajoutez la virgule manquante ici
                }
            )

    return markers


def is_subtree(tree1, tree2):
    """
    Vérifie si l'arbre tree2 est un sous-arbre de l'arbre tree1.
    """

    def convert_to_graph(tree):
        G = nx.DiGraph()
        for region_id, region in tree.items():
            G.add_node(region_id)
            if region.parent is not None:
                G.add_edge(region.parent, region_id)
        return G

    graph1 = convert_to_graph(tree1)
    graph2 = convert_to_graph(tree2)

    matcher = nx.isomorphism.DiGraphMatcher(graph1, graph2)

    mapping = next(matcher.subgraph_isomorphisms_iter(), None)

    return mapping


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


def visualize_markers(image, marker_data):
    """
    Visualise les marqueurs avec leurs positions et couleurs associées.

    :param marker_data: Liste de dictionnaires contenant les données des marqueurs.
    """

    color_mapping = {
        1: [1, 0, 0],  # Rouge
        2: [0, 1, 0],  # Vert
        3: [0, 0, 1],  # Bleu
        4: [1, 1, 0],  # Jaune
        5: [1, 0, 1],  # Magenta
    }

    # Convert grayscale image to RGB
    image[image == 1] = 255

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for marker in marker_data:
        marker_id = marker["marker_id"]
        position = tuple(map(int, marker["position"]))
        color = color_mapping.get(marker_id, [1, 1, 1])  # Blanc par défaut

        image_rgb[position] = np.array(color) * 255  # Scale back to 0-255 range

        for neighbor_position in get_neighbors(position, 10):
            x, y = map(int, neighbor_position)
            image_rgb[x, y] = np.array(color) * 255

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

    plt.show()


# def visualize_markers(image, marker_data):
#     """
#     Visualise les marqueurs avec leurs positions et couleurs associées.

#     :param marker_data: Liste de dictionnaires contenant les données des marqueurs.
#     """

#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     color_mapping = {
#         1: [255, 0, 0],  # Rouge
#         2: [0, 255, 0],  # Vert
#         3: [0, 0, 255],  # Bleu
#         4: [255, 255, 0],  # Jaune
#         5: [255, 0, 255],  # Magenta
#     }

#     for marker in marker_data:
#         marker_id = marker["marker_id"]
#         position = tuple(map(int, marker["position"]))
#         color = color_mapping.get(marker_id, [255, 255, 255])  # Blanc par défaut

#         image[position] = color

#     # Afficher l'image
#     plt.imshow(image)
#     plt.title("Visualisation des marqueurs")

#     # Créer une légende
#     legend_labels = [f"Marker {marker_id}" for marker_id in color_mapping.keys()]
#     legend_colors = list(color_mapping.values())

#     plt.legend(
#         handles=[
#             plt.Line2D(
#                 [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
#             )
#             for color in legend_colors
#         ],
#         labels=legend_labels,
#         loc="upper left",
#         bbox_to_anchor=(1, 1),
#     )

#     plt.show()
