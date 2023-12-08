import numpy as np
import json
import os


def load_trees(dir):
    trees = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".json"):
            adjacency_tree = {}
            file_path = os.path.join(dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for key, region_data in data.items():
                    color = region_data["color"]
                    parent = region_data["parent"]
                    barycentre = region_data["barycentre"]
                    children = region_data["children"]
                    pixel_region = PixelRegion(color, parent, barycentre, children)
                    key = int(key)
                    adjacency_tree[key] = pixel_region
            trees.append(adjacency_tree)
    return trees


def save_adjacency_tree_to_json(adjacency_tree, output_path):
    """
    Enregistre l'arbre d'adjacence dans un fichier JSON.

    :param adjacency_tree: Arbre d'adjacence.
    :param output_path: path pour sauvegarder le json.
    """

    json_data = {}
    for region_id, region in adjacency_tree.items():
        json_data[region_id] = region.to_dict()

    # Écrire les données JSON dans le fichier
    with open(output_path, "w") as json_file:
        json.dump(
            json_data,
            json_file,
            indent=2,
            ensure_ascii=False,
            default=convert_to_json_serializable,
        )


class PixelRegion:
    """
    Initialise un objet PixelRegion avec la couleur spécifiée et une référence optionnelle à un parent.

    :param color: Couleur du pixel dans la région.
    :param parent: Référence optionnelle à la région parente.
    """

    def __init__(self, color, parent=None, barycentre=None, children=None):
        self.color = color
        self.parent = parent
        self.children = children if children is not None else []
        self.barycentre = barycentre
        self.pixel_count = 0

    def add_child(self, child):
        self.children.append(child)

    def update_barycentre(self, x, y):
        if self.pixel_count == 0:
            self.barycentre = (x, y)
        else:
            # Mise à jour incrémentale du barycentre
            self.barycentre = (
                (self.barycentre[0] * self.pixel_count + x) / (self.pixel_count + 1),
                (self.barycentre[1] * self.pixel_count + y) / (self.pixel_count + 1),
            )
        self.pixel_count += 1

    def get_barycentre(self):
        return self.barycentre

    def get_parent(self):
        return self.parent

    def get_color(self):
        return self.color

    def get_children(self):
        return self.children

    def set_parent(self, parent):
        self.parent = parent

    def to_dict(self):
        region_dict = {
            "color": self.color,
            "barycentre": self.barycentre,
            "parent": self.parent,
            "children": self.children,
        }

        return region_dict


def flood_fill(
    image,
    start_x,
    start_y,
    current_color,
    region,
    visited_color,
    adjacency_tree,
    image_regions,
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
    :param image_regions: Matrice représentant les régions (pour visualisation).
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
                if region == 2:
                    parent_id = 1
                    adjacency_tree[region].set_parent(parent_id)
                    if region not in adjacency_tree[parent_id].get_children():
                        adjacency_tree[parent_id].add_child(region)
                else:
                    if adjacency_tree[region].get_parent() == None:
                        # Mettre à jour la relation parent-enfant si le pixel est déjà visité
                        parent_id = image_regions[x, y]
                        adjacency_tree[region].set_parent(parent_id)
                        if region not in adjacency_tree[parent_id].get_children():
                            adjacency_tree[parent_id].add_child(region)
            continue

        # Ignorer les pixels qui ne sont pas de la couleur actuelle
        if image[x, y] != current_color:
            continue

        # Marquer le pixel comme visité
        image[x, y] = np.int_(-1)
        image_regions[x, y] = region
        adjacency_tree[region].update_barycentre(x, y)

        # Ajouter le pixel
        cpt += 1
        if cpt % 1000000 == 0:
            print(cpt)

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

    image_regions = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= 0:  # pixel non visité
                # initialisation de la nouvelle région
                adjacency_tree[region_counter] = PixelRegion(image[i, j])

                # remplissage de la région
                flood_fill(
                    image,
                    i,
                    j,
                    image[i, j],
                    region_counter,
                    -1,
                    adjacency_tree,
                    image_regions,
                )

                # fin de la région actuelle
                region_counter += 1

    return adjacency_tree, image_regions


def convert_to_json_serializable(obj):
    """
    Convertit les objets non sérialisables en des types natifs JSON sérialisables.
    """
    if isinstance(obj, (np.int64, np.uint8)):
        return int(obj)
    else:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )
