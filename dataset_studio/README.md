# Dataset Studio

Interface Qt pour **voir, contrôler et modifier** les enregistrements de la téléop du G1-D, avant de les convertir au format WLA. Thème repris du cockpit de mpc_any.

## Lancer

Env `g1d_teleop` (PySide6 y est installé ; `teleoperation/setup_env.sh` l'installe aussi) :

```bash
conda activate g1d_teleop
~/Documents/project/manip/unifolm-wla/studio.sh
```

`studio.sh` marche depuis n'importe quel dossier. Sans argument, il ouvre le dossier d'enregistrement de la téléop : `teleoperation/Tele_OP/xr_teleoperate/teleop/utils/data`. Le menu **Tâche** liste alors chaque dataset (`novares_box`, `napkin`…).

Pour ouvrir un autre dossier, une tâche ou un dossier de tâches :

```bash
~/Documents/project/manip/unifolm-wla/studio.sh /chemin/vers/utils/data/napkin
```

Équivalent sans le script, depuis la racine du dépôt : `python -m dataset_studio [dossier]`.

## L'écran

| Zone | Contenu |
|---|---|
| **Gauche — Épisodes** | Liste des épisodes : durée, nombre de pas, issue (réussi / raté / inconnu), pinces utilisées (● gauche / ● droite), indicateurs (base ou colonne en mouvement, images manquantes, épisode de moins d'1 s, JSON illisible). En haut : total d'épisodes, de pas et de minutes. |
| **Centre — Caméras** | Les images synchronisées : tête œil gauche et œil droit, poignet gauche, poignet droit (3 vues pour la sim). Barre de lecture, vitesse ×0,25 à ×4. |
| **Centre — Signaux** | Deux graphiques, chacun avec son groupe de courbes : bras gauche ou droit (état mesuré, consigne), pinces (état, consigne), rotation du buste, colonne (hauteur, consigne), base (vx, vyaw), intervention / avantage en mode politique. Le curseur suit la lecture ; cliquer ou glisser dans un graphique déplace la lecture. |
| **Droite — Épisode** | Consigne (instruction du modèle), issue de l'essai, rognage, découpage en sous-tâches, en-tête `info` du fichier, journal des actions. |

## Raccourcis

| Touche | Effet |
|---|---|
| Espace | Lecture / pause |
| ← / → | Pas précédent / suivant |
| Maj + ← / → | Recul / avance d'1 s |
| ↑ / ↓ | Épisode précédent / suivant |
| I / O | Début / fin au pas courant (rognage ou segment) |
| 1 à 9 | Choisir l'étiquette n° 1 à 9 |
| T | Étiqueter [début, fin] avec l'étiquette choisie |
| Suppr | Supprimer les épisodes sélectionnés |

Ctrl ou Maj + clic dans la liste sélectionne plusieurs épisodes.

## Modifier un dataset

Toutes les modifications destructives passent par la **corbeille** `<tâche>/.corbeille/`, ignorée par l'enregistreur et par le convertisseur.

- **Supprimer** (🗑 ou Suppr) : les épisodes sélectionnés vont dans la corbeille, puis les suivants sont **renumérotés sans trou** (`episode_0000`, `0001`…).
- **Corbeille…** : restaurer un épisode supprimé. Il revient **à la fin** du dataset, avec un nouveau numéro. « Vider la corbeille » supprime définitivement les épisodes et les sauvegardes de rognage.
- **Rogner** : placez le début (**I**) et la fin (**O**), puis « ✂ Rogner l'épisode ». Seuls les pas entre les deux sont gardés. L'ancien `data.json` et les images retirées sont rangés dans la corbeille (`…__rognage__episode_XXXX`). `success_step` et `takeover_step` sont recalés.
- **Consigne** : modifiez le texte, puis appliquez-le à l'épisode courant, à la sélection ou à tous les épisodes.
- **Issue** : réussi / raté / inconnu / non renseignée. Passer à « réussi » fixe `success_step` au dernier pas s'il est absent.
- **Renuméroter** : remet des numéros contigus sans rien supprimer, par exemple si un dataset commence à `episode_0001`.

## Découper en sous-tâches

Pour entraîner des politiques plus simples (par exemple « prise main gauche », « prise main droite », « empiler »), on découpe chaque épisode en segments étiquetés, puis on exporte un sous-dataset.

1. **Créer les étiquettes** (panneau *Découpage en sous-tâches*, bouton **＋**) : un nom sans espace (`prise_gauche`) et la **consigne du modèle** pour cette sous-tâche (`pick up the black object with the left hand`). **✎** renomme l'étiquette ou modifie sa consigne (ses segments suivent), **−** supprime l'étiquette, avec tous ses segments après confirmation.
2. **Découper** : placez le début (**I**), avancez jusqu'à la fin de la sous-tâche, choisissez l'étiquette (**1** à **9**) puis **T**. Sans fin posée, le segment s'arrête au pas courant. Le début du segment suivant est placé juste après : on enchaîne **I** une fois, puis **T** à chaque changement de sous-tâche.
3. Les segments apparaissent en bandes colorées sous les graphiques et dans la liste du panneau. Double-clic : aller au segment (et reprendre ses bornes). Pour effacer un segment raté : cliquez dessus dans la liste, puis **Supprimer le segment** ou **Suppr** (Suppr ne supprime un épisode que si le focus n'est pas sur la liste des segments).
4. **Exporter** :
   - **Exporter par étiquette…** crée un dossier de tâche par étiquette, `<tâche>__<étiquette>` (une politique par sous-tâche) ;
   - **Exporter tout…** crée un seul dossier `<tâche>__segments`, chaque segment avec la consigne de son étiquette (une seule politique qui suit la consigne).

Les exports sont créés à côté de la tâche, apparaissent dans le menu **Tâche** et se convertissent avec **Convertir au format WLA** comme les autres. Chaque épisode exporté est un segment ; son `info.segment` dit d'où il vient. Les images sont des liens physiques : presque aucune place disque en plus. Un nouvel export **remplace** l'export précédent du même nom (jamais un dossier qui n'est pas un export).

Stockage : `<tâche>/tags.json` (étiquettes) et `episode_XXXX/segments.json` (segments, pas inclus). Les segments suivent les épisodes renumérotés ou mis à la corbeille, et un rognage les décale.

## Convertir au format WLA

Le bouton **Convertir au format WLA** lance `g1d_wla.convert_teleop` sur la tâche affichée, avec l'env `.venv` du dépôt. Le résultat va dans :

```
playground/Datasets/g1d_<tâche>/<tâche>
```

La progression s'affiche dans le journal. Un dataset déjà présent à cet endroit est remplacé.

**Après toute suppression ou tout rognage, il faut reconvertir** avant de relancer un entraînement. Un entraînement déjà lancé utilise la version convertie à son démarrage.

Pour entraîner ensuite, voir `sim/experiments/train_real.sh` (config de données dans `unifolm_wla/dataloader/multi_source_dataset/configs/g1d_<tâche>.yaml`, copie de `g1d.yaml` avec le bon `data_path`).

## Fichiers

| Fichier | Rôle |
|---|---|
| `model.py` | Couche données, sans Qt : lecture, résumé, courbes, suppression / corbeille / renumérotation, rognage, consigne, issue. Testable seule. |
| `app.py` | Fenêtre principale (PySide6) |
| `charts.py` | Graphique QPainter : séries, curseur, zone de rognage |
| `theme.py` | Couleurs et feuille de style |
