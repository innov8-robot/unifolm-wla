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
| **En-tête** | Tâche (menu), résumé (épisodes, minutes, réussis / ratés / sans résultat), **Objets…**, Recharger, Ouvrir…, **? Aide** (F1 ou ?) |
| **Gauche — Épisodes** | Filtre (tous, sans résultat, réussis, ratés, non découpés, à vérifier) et recherche par numéro. Colonnes : n°, durée, résultat (✓ ✗ ?), état (▦ n segments, ◎ objets suivis, ⚠ à vérifier). Le détail (pas, pinces, indicateurs) est dans l'infobulle. |
| **Centre — Vidéo** | La caméra de **tête en grand** (œil gauche, celle que voit le modèle) et les deux poignets dessous. Barre de lecture, vitesse ×0,25 à ×4. |
| **Centre — Frise** | Sous la vidéo : segments colorés, sélection, pas corrigés par l'opérateur (tirets jaunes), curseur. Clic = aller à ce pas, glisser = sélectionner une plage, clic droit sur un segment = supprimer / reprendre ses bornes. |
| **Centre — Signaux** | Les pinces par défaut. **＋ 2ᵉ graphique** pour un second groupe (angles des bras, buste, colonne, base, intervention). |
| **Droite** | Sections repliables : **Épisode** (résultat en un clic, consigne), **Sélection sur la frise** (étiqueter ou rogner), **Sous-tâches**, **Détails du fichier**, **Journal**. |
| **Bas** | Barre de messages : chaque action y est confirmée. |

## Raccourcis

| Touche | Effet |
|---|---|
| Espace | Lecture / pause |
| ← / → · Maj + ← / → | Pas précédent / suivant · ± 1 s |
| ↑ / ↓ | Épisode précédent / suivant |
| I / O · Échap | Début / fin de la sélection · effacer la sélection |
| **R / E / N** | Résultat : réussi / raté / inconnu (re-appuyer = effacer) |
| 1 à 9 · T | Choisir l'étiquette · étiqueter la sélection |
| Suppr | Supprimer les épisodes sélectionnés (ou le segment sélectionné dans la liste des segments) |
| F1 ou ? | Aide |

Ctrl ou Maj + clic dans la liste sélectionne plusieurs épisodes. Les touches ne marchent pas pendant la saisie dans un champ de texte.

Ouverture rapide : les résumés des épisodes sont gardés dans `<tâche>/.studio_cache.json` (relu seulement pour les épisodes modifiés).

## Modifier un dataset

Toutes les modifications destructives passent par la **corbeille** `<tâche>/.corbeille/`, ignorée par l'enregistreur et par le convertisseur.

- **Supprimer** (🗑 ou Suppr) : les épisodes sélectionnés vont dans la corbeille, puis les suivants sont **renumérotés sans trou** (`episode_0000`, `0001`…).
- **Corbeille…** : restaurer un épisode supprimé. Il revient **à la fin** du dataset, avec un nouveau numéro. « Vider la corbeille » supprime définitivement les épisodes et les sauvegardes de rognage.
- **Rogner** : sélectionnez la plage à garder (glisser sur la frise, ou **I** / **O**), puis « ✂ Rogner l'épisode ». Seuls les pas entre les deux sont gardés. L'ancien `data.json` et les images retirées sont rangés dans la corbeille (`…__rognage__episode_XXXX`). `success_step` et `takeover_step` sont recalés.
- **Consigne** : modifiez le texte, puis appliquez-le à l'épisode courant, à la sélection ou à tous les épisodes.
- **Résultat** : boutons ✓ / ✗ / ? ou touches R / E / N, appliqué tout de suite (re-cliquer le résultat actif l'efface). « Réussi » fixe `success_step` au dernier pas s'il est absent.
- **Renuméroter** : remet des numéros contigus sans rien supprimer, par exemple si un dataset commence à `episode_0001`.

## Découper en sous-tâches

Pour entraîner des politiques plus simples (par exemple « prise main gauche », « prise main droite », « empiler »), on découpe chaque épisode en segments étiquetés, puis on exporte un sous-dataset.

1. **Créer les étiquettes** (panneau *Découpage en sous-tâches*, bouton **＋**) : un nom sans espace (`prise_gauche`) et la **consigne du modèle** pour cette sous-tâche (`pick up the black object with the left hand`). **✎** renomme l'étiquette ou modifie sa consigne (ses segments suivent), **−** supprime l'étiquette, avec tous ses segments après confirmation.
2. **Découper** : placez le début (**I**), avancez jusqu'à la fin de la sous-tâche, choisissez l'étiquette (**1** à **9**) puis **T**. Sans fin posée, le segment s'arrête au pas courant. Le début du segment suivant est placé juste après : on enchaîne **I** une fois, puis **T** à chaque changement de sous-tâche.
3. Les segments apparaissent en bandes colorées sous les graphiques et dans la liste du panneau. Double-clic : aller au segment (et reprendre ses bornes). Pour effacer un segment mal placé : **clic droit sur sa bande** dans la frise (graphiques), puis « Supprimer le segment » ; « Reprendre ses bornes » remet son début et sa fin en I / O. Ou cliquez dessus dans la liste, puis **Supprimer le segment** ou **Suppr** (Suppr ne supprime un épisode que si le focus n'est pas sur la liste des segments).
4. **Exporter** :
   - **Exporter par étiquette…** crée un dossier de tâche par étiquette, `<tâche>__<étiquette>` (une politique par sous-tâche) ;
   - **Exporter tout…** crée un seul dossier `<tâche>__segments`, chaque segment avec la consigne de son étiquette (une seule politique qui suit la consigne).

Les exports sont créés à côté de la tâche, apparaissent dans le menu **Tâche** et se convertissent avec **Convertir au format WLA** comme les autres. Chaque épisode exporté est un segment ; son `info.segment` dit d'où il vient. Les images sont des liens physiques : presque aucune place disque en plus. Un nouvel export **remplace** l'export précédent du même nom (jamais un dossier qui n'est pas un export).

Stockage : `<tâche>/tags.json` (étiquettes) et `episode_XXXX/segments.json` (segments, pas inclus). Les segments suivent les épisodes renumérotés ou mis à la corbeille, et un rognage les décale.

## Étiqueter « bon » / « mauvais » pour l'apprentissage par renforcement (RECAP)

Sur un dataset d'essais en mode politique (`…_policy`), le bouton **＋ bon / mauvais (RL)** crée deux étiquettes : `bon` (vert) et `mauvais` (rouge). Découpez les passages comme pour les sous-tâches (I, puis **1** T pour bon, **2** T pour mauvais).

Lors de l'étiquetage RECAP (`g1d_wla.recap label`), ces passages **remplacent** le jugement automatique du modèle de valeur et les corrections au grip : `bon` = avantage positif, `mauvais` = négatif. Les pas sans étiquette gardent l'étiquette automatique. Exemple : une pile de 3 où la 3ᵉ pièce tombe → `bon` sur les deux premières poses, `mauvais` sur la 3ᵉ.

Après un étiquetage RECAP, le graphique « Intervention / avantage » montre l'étiquette calculée par pas (1 positif, 0 négatif) : c'est là qu'on voit où corriger.

## Objets : encadrer, détecter, choisir source / cible, suivre (optionnel)

Bouton **Objets…** de l'en-tête. Fonction facultative : sans elle, rien ne change. Les calculs (DINOv3, SAM2) tournent dans un autre environnement, par défaut `~/miniconda3/envs/unitree_lerobot` (à changer avec la variable `STUDIO_OBJECTS_PY`) ; les poids sont lus dans `mpc_any/checkpoints` (variable `STUDIO_OBJECTS_CKPT`). La fenêtre suit l'épisode et l'image du studio (caméra de tête).

1. **Encadrer** : glissez un rectangle autour de l'objet, puis **Ajouter l'encadré**. 10 à 20 encadrés sur des images variées (**Image au hasard**) : objet posé, vu sous plusieurs angles, dans la pince.
2. **Construire la signature** : le journal donne la validation (chaque exemple retiré à tour de rôle doit rester au-dessus de 0, les fonds en dessous).
3. **Détecter** : sur une image où la main est loin des pièces, **Détecter ici** (ou **Tous les épisodes**, à l'image 0). SAM2 découpe tout, la signature garde ce qui ressemble à l'objet : 6 pièces identiques donnent 6 masques numérotés. Aucun texte.
4. **Choisir** : un clic sur un masque = **source** (vert), un clic sur un autre = **cible** (rouge), nouveau clic = retirer. Les autres masques sont écartés. **Enregistrer le choix**. Pour un second empilement dans le même épisode : aller plus loin dans la vidéo, détecter, choisir, enregistrer ; chaque choix vaut jusqu'au suivant.
5. **Suivre** (cet épisode, ou tous ceux qui ont un choix) : seuls les deux masques choisis sont suivis, vers l'avant (comme en direct sur le robot). La fenêtre les affiche ensuite image par image.

Fichiers : `objects/objects.json` et `objects/<objet>_signature.npz` dans la tâche ; `episode_XXXX/objects/` (candidats, choix, masques suivis). Ligne de commande : `python -m dataset_studio.objects_worker --help`.

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
| `model.py` | Couche données, sans Qt : lecture, résumé, courbes, suppression / corbeille / renumérotation, rognage, consigne, résultat, étiquettes et segments, cache des résumés. Testable seule. |
| `timeline.py` | Frise sous la vidéo : segments, sélection, interventions, curseur |
| `objects_ui.py`, `objects_worker.py` | Fenêtre Objets (optionnelle) et ses calculs, lancés dans un autre environnement |
| `app.py` | Fenêtre principale (PySide6) |
| `charts.py` | Graphique QPainter : séries, curseur, zone de rognage |
| `theme.py` | Couleurs et feuille de style |
