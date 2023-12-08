## TP2 - IMN502

Julien Bernat - berj0505
Florence Gagnon - gagf3801

## Usage

Le projet soumis a déjà été roulé, et les résultats sont enregistrés dans les dossiers.

Pour rouler le projet au complet, il est possible de rouler (dans l'ordre) :

- python compute_markers.py

Ce fichier permet de créer les arbres d'adjacence des marqueurs. Ils seront enregistrés sous forme de json dans lirairy_trees/.

- python compute_images.py

Ce fichier permet de créer les arbres d'adjacence des images de détection. Ils seront enregistrés sous forme de json dans detection_trees/.

- python compute_detection.py

Ce fichier permet ensuite de détecter les marqueurs dans les images de détection. Le output sera enregistré dans resultats/ (resultats.txt).

## Dossiers

- /librairie2 : dossier de base contenant les marqueurs
- /detection : dossier de base contenant les images de détection
- /detection_trees : dossier contenant les arbres d'adjacence des images de détection sous forme de json
- /librairy_trees : dossier contenant les arbres d'adjacence des marqueurs sous forme de json
- /resultats : dossier contenant les figures et les résultats
