import numpy as np
import networkx as nx


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
        mapping = is_subtree(detection_tree, markers_tree)

        if mapping is not None and len(mapping) > 1:
            keys_list = list(mapping.keys())
            second_element_key = keys_list[1]

            # Calcul de l'orientation
            barycentre_black = np.array([0, 0])
            barycentre_white = np.array([0, 0])
            count_black = 0
            count_white = 0
            first_color = 0
            cpt = 0

            for key in detection_tree[keys_list[2]].get_children():
                if cpt == 0:
                    first_color = detection_tree[key].get_color()
                color = detection_tree[key].get_color()
                barycentre = np.array(detection_tree[key].get_barycentre(), dtype=int)
                if detection_tree[key].get_children() == []:
                    if color == 0:  # Noir
                        barycentre_black += barycentre
                        count_black += 1
                    elif color == 1:  # Blanc
                        barycentre_white += barycentre
                        count_white += 1
                else:
                    for key_child in detection_tree[key].get_children():
                        color = detection_tree[key_child].get_color()
                        barycentre = np.array(
                            detection_tree[key_child].get_barycentre(), dtype=int
                        )
                        if color == 0:  # Noir
                            barycentre_black += barycentre
                            count_black += 1
                        elif color == 1:  # Blanc
                            barycentre_white += barycentre
                            count_white += 1
                cpt += 1

            if first_color == 0:  # Noir
                orientation = np.array(barycentre_black / count_black, dtype=int)
            else:
                orientation = np.array(barycentre_white / count_white, dtype=int)

            # Calcul de la position
            position = detection_tree[second_element_key].get_barycentre()

            markers.append(
                {
                    "marker_id": i,
                    "position": position,
                    "orientation": orientation,
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

    def compare_subtrees(graph1, node1, graph2, node2, mirror=False):
        children1 = list(graph1.successors(node1))
        children2 = list(graph2.successors(node2))

        if len(children1) != len(children2):
            return False

        if mirror:
            # Comparer les enfants dans l'ordre inverse
            children2 = reversed(children2)

        for child1, child2 in zip(children1, children2):
            if not compare_subtrees(graph1, child1, graph2, child2, mirror):
                return False

        return True

    graph1 = convert_to_graph(tree1)
    graph2 = convert_to_graph(tree2)

    matcher = nx.isomorphism.DiGraphMatcher(graph1, graph2)

    mapping = next(matcher.subgraph_isomorphisms_iter(), None)

    # partie à commenter pour avoir un matching moins stricte
    if mapping:
        for i, node2 in enumerate(mapping.values()):
            for j, node1 in enumerate(mapping):
                if i == 2 and j == 2:
                    # Comparer la version normale
                    if compare_subtrees(graph1, node1, graph2, node2):
                        return mapping

                    # Comparer la version miroir
                    if compare_subtrees(graph1, node1, graph2, node2, mirror=True):
                        return mapping

    return None
