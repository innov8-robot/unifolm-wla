# Inference Studio

Application Qt pour **essayer un modèle sur le G1-D** sans terminal : choisir le modèle, démarrer le serveur, lancer la téléop en mode politique, puis piloter les essais avec de gros boutons ou le clavier.

## Lancer

```bash
~/Documents/project/manip/unifolm-wla/inference.sh
```

(Active l'env `g1d_teleop`, dont les variables pointent les certificats du casque.)

## Déroulé

1. **Modèle** : la liste montre les modèles entraînés (`playground/Checkpoints/*/final_model`), le plus récent en haut. La **consigne** propose celles des données d'entraînement du modèle (modifiable). Options : enregistrer les essais (dossier `<modèle>_policy`), rotation du buste, base, colonne, vitesse max.
2. **Démarrer le serveur** : charge le modèle sur le GPU (≈ 1 min 20) ; pastille « Serveur prêt ».
3. **Lancer la téléop** : mode politique avec la consigne choisie. Le casque est optionnel (`adb reverse` est fait s'il est branché) : sans casque, pas de correction au grip, tout le reste marche.
4. **Activer le robot** (Entrée), puis les essais :

| Bouton | Touche | Effet (= manette) |
|---|---|---|
| Activer le robot | Entrée | « r » : les bras passent sous contrôle |
| Lancer l'essai / Arrêter | Espace | A droit : le modèle prend la main (arrêter sans résultat = issue inconnue) |
| Réussi | R | X gauche |
| Raté | E | Y gauche |
| Annuler | Échap | B droit : essai non enregistré |
| Gravité zéro / Tenir les bras | Z | clic joystick gauche (hors essai) |
| Garde | G | Y gauche hors essai |
| Quitter la téléop | Q | « q » : bras ramenés au repos lentement |

Pastilles en haut : serveur, téléop, robot, essai en cours (nombre de pas), gravité zéro, caméras (absente / tête floue), réussis / ratés de la session. Journaux du serveur et de la téléop à droite.

**Arrêt d'urgence : le bouton physique du robot.** La manette et le casque restent utilisables en même temps que l'application.

Les derniers choix (modèle, consigne par modèle, options) sont gardés dans `~/.config/g1d_inference_studio.json`.

## Visual prompt (modèles entraînés avec les couleurs, ex. `g1d_novares_vp`)

Cocher **Visual prompt** (coché tout seul pour un modèle dont le nom contient `_vp`) et choisir le projet dont la **signature** reconnaît les pièces. Le service de suivi (`inference_studio/vp_tracker.py`, env `unitree_lerobot`) démarre avec le serveur ; la vue de la caméra de tête apparaît au centre.

1. Bras **loin des pièces** (garde) : **D** = détecter les pièces (numérotées de gauche à droite).
2. **1 à 9** : la pièce à PRENDRE (verte), puis la DESTINATION (rouge). **0** = recommencer.
3. La pastille « VP : suivi ✓ » passe au vert : **Espace** lance l'essai. Le modèle reçoit l'image de tête avec le même rendu vert / rouge qu'à l'entraînement (`dataset_studio/visual_prompt.py`).

Tant que le suivi n'est pas prêt, la téléop ne demande rien au modèle : le robot attend. Le suivi est fait image par image (SAM2 relancé avec la boîte de l'image précédente, méthode de mpc_any) ; un masque qui grossit d'un coup (il « avale » la pince) est refusé et le précédent est gardé.

## Comment ça marche

La téléop est lancée avec `--ipc` : l'application lui envoie des commandes (`CMD_START`, `CMD_TRIAL`, `CMD_SUCCESS`, `CMD_FAILURE`, `CMD_CANCEL`, `CMD_ZERO_G`, `CMD_GUARD`, `CMD_STOP`, voir `teleop/utils/ipc.py`) et lit son état publié dix fois par seconde (essai en cours, pas, gravité zéro, caméras, netteté, réussis / ratés). Les commandes ont exactement le même effet que les boutons de la manette.
