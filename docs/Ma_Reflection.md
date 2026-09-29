# Briefing — Portage de UnifoLM-WLA-1.0 sur Unitree G1-D

**Date du briefing :** 28 septembre 2026
**Destinataire :** agent de développement travaillant sur ce projet
**Objectif du projet :** faire tourner UnifoLM-WLA-1.0 sur un Unitree G1-D pour des tâches de manipulation

---

## 0. Comment lire ce document

Ce briefing distingue trois niveaux de confiance. Respecter cette distinction : ne pas traiter une inférence comme un fait établi.

- **[VÉRIFIÉ]** — constaté directement dans les fichiers officiels (README, docs du repo, config.json des modèles, listings Hugging Face).
- **[INFÉRÉ]** — déduction cohérente mais non confirmée par une source. À valider avant de s'appuyer dessus.
- **[INCONNU]** — question ouverte, à résoudre en lisant le code ou en testant.

---

## 1. État de la release

**[VÉRIFIÉ]** Au 28 septembre 2026, tout est publié sauf le LoRA :

| Élément | Statut |
|---|---|
| UnifoLM-ER-1 (VLM raisonneur) | publié 11 sept. |
| UnifoLM-ER-Flow (backbone) | publié 11 sept. |
| Code d'entraînement action expert | publié 20 sept. |
| **UnifoLM-WLA-1.0-Base + code de fine-tuning** | **publié 28 sept.** |
| UniBot-V1 Challenge Dataset | publié |
| UnifoLM-WBT-Dataset (whole-body) | publié |
| UnifoLM-Dex1-Dataset (G1 + gripper 2 doigts) | publié |
| Code de fine-tuning LoRA | **non publié** |

Conséquence : le fine-tuning complet de l'action expert est disponible, le LoRA non. Ce n'est pas bloquant — la recette officielle gèle déjà le VLM (voir §3).

**Repos concernés :**
- `github.com/unitreerobotics/unifolm-wla` — le modèle WLA, son code d'entraînement et de fine-tuning
- `github.com/unitreerobotics/unifolm-world-model-action` — génération précédente (WMA-0), contient **`unitree_deploy`**, la couche de déploiement robot réel
- `github.com/unitreerobotics/unitree_lerobot` — conversion de données, entraînement LeRobot, déploiement
- `github.com/unitreerobotics/xr_teleoperate` — téléopération VR avec enregistrement

---

## 2. Matériel cible

**[VÉRIFIÉ]** Unitree G1-D :
- 17 DOF (Standard, base fixe) ou 19 DOF (Flagship/Ultimate, châssis différentiel à roues, 1,5 m/s)
- 7 DOF par bras, charge 3 kg, portée 0,45 m
- Colonne élévatrice 500 mm (hauteur totale 1260–1680 mm, hauteur de travail jusqu'à 2 m)
- Caméra binoculaire de tête 3840×1200, caméras de poignet 1920×1080
- Jetson Orin NX embarqué (100 TOPS)
- Extensible jusqu'à 31 DOF avec mains dextres

**Différence avec la plateforme d'entraînement :** WLA-1.0 a été entraîné sur le **G1 bipède** (jusqu'à ~43 DOF corps entier). Le G1-D n'a pas de jambes. Voir §5.

---

## 3. Recette de fine-tuning officielle

**[VÉRIFIÉ]** Script : `examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh`

Le backbone VLM est **gelé** (`trainer.freeze_modules: qwen_vl_interface`). Seuls s'entraînent :
1. la tête action expert (DiT / MMDiT)
2. **le robot-state projector**

Ce second point est important pour ce projet. Le robot-state projector est la couche qui mappe l'état robot dans l'espace latent du VLM (hidden_size 2560). C'est donc **l'adaptateur d'embodiment naturel** : c'est la pièce qui doit ré-apprendre pour passer du G1 au G1-D.

Pour co-entraîner le VLM (plus de VRAM) : vider `trainer.freeze_modules`.

Config à éditer : `unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml`

Lancement :
```bash
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
bash examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh
```

---

## 4. ⚠️ Piège principal : le serveur est Dex1-only

**[VÉRIFIÉ]** — documenté explicitement dans `docs/train_action_expert_en.md`.

`_ACTIVE_SLOT_SPECS` dans `model_server/unifolm_wla_action_adapter.py` ne couvre que l'espace d'action **Dex1 (gripper deux doigts)**. Il n'a **aucune entrée** pour :
- `left_fig6d` / `right_fig6d` — angles de doigts des mains dextres
- `base_pose` — mouvement relatif de base (whole-body)

Ces champs n'existent que sous `unitree_fullbody_base` (WBT). Le protocole obs/action du serveur (`_DATA_KEYS`, `_build_state_unnorm`, `_encode_action` dans `action_server_wbc_msgpack_unitree.py`) est **hardcodé** et ne les transporte pas.

### Ce que ça implique selon l'effecteur

| Effecteur | Travail requis |
|---|---|
| **Dex1 (gripper 2 doigts)** | Aucun patch. Chemin nominal. |
| **Dex3-1 / main 5 doigts** | Patcher `_ACTIVE_SLOT_SPECS` (ajouter `left_fig6d`, `right_fig6d`) **et** étendre le protocole obs/action du serveur. |

Si le châssis roulant du G1-D Ultimate doit être piloté par le modèle, il faut aussi ajouter `base` / `base_rotvec` — même travail.

**➜ Question bloquante à résoudre en premier : quel effecteur équipe ce G1-D ?**

---

## 5. Transfert cross-embodiment G1 → G1-D

### Ce qui joue en notre faveur

**[VÉRIFIÉ]** Sur les 64 tâches d'entraînement, **54 sont du tabletop** et 10 seulement du whole-body. Les tâches tabletop ont nécessairement les slots jambes à masque 0. Masquer le bas du corps n'est donc pas un détournement : c'est un mode nominal, vu massivement à l'entraînement.

**[VÉRIFIÉ]** Les bras sont pilotés en **espace tâche**, pas en espace articulaire : l'espace d'action unifié place en `[0:6]` et `[13:19]` des poses SE(3) relatives de l'effecteur. Le modèle sort un Δpose ; l'IK locale traduit en commandes moteur. Des cinématiques de bras légèrement différentes sont absorbées par cette couche.

**[VÉRIFIÉ]** Le transfert *réduit* l'espace d'action (suppression de la locomotion bipède), ce qui est le cas favorable.

### Ce qui reste à régler

**[INFÉRÉ]** Trois recalages, par ordre d'impact estimé :

1. **Repère base `B`.** L'état effecteur `[0:9]` / `[16:25]` est une pose *absolue* dans le repère base. Sur G1 ce repère est au bassin d'un bipède ; sur G1-D il faut le définir. **Le poser au point du torse correspondant au bassin du G1 à hauteur alignée** pour que les poses absolues tombent dans la même distribution et que les statistiques de normalisation restent valides. C'est une transformation fixe, à déterminer une fois depuis les deux URDF.

2. **Hauteur et pitch de la tête.** La colonne élévatrice (500 mm de course) permet d'aligner la hauteur de tête sur celle d'un G1 debout. Vérifier aussi l'angle de plongée de la caméra sur la table — c'est ce qui change le plus l'image.

3. **Géométrie épaules ↔ caméra.** Si les épaules ne sont pas à la même position relative à la caméra que sur un G1, les bras apparaissent ailleurs dans l'image. Mesurer sur les deux URDF ; si l'écart est de quelques centimètres, ignorer.

**[INCONNU]** Les caméras. Les fiches produit donnent une RealSense D435i pour le G1 catalogue et une binoculaire 3840×1200 / 115° pour le G1-D — mais les vidéos de démo WLA ne montrent pas de D435i, ce qui suggère qu'Unitree a utilisé un rig de collecte dédié, potentiellement le même module. **À trancher** en lisant les intrinsèques/résolution attendues dans la config de preprocessing et les métadonnées caméra des datasets publiés. Si un écart de FOV subsiste, il se rattrape par recadrage/ré-échantillonnage du flux (calibration, pas fine-tune).

**[INFÉRÉ]** Coût en données estimé : un fine-tuning léger (dizaines à quelques centaines de démos) devrait suffire, puisque la partie manipulation est quasi identique. Ne **pas** partir sur un ré-entraînement complet avant d'avoir testé le zero-shot puis le fine-tune léger.

---

## 6. Spécification des espaces unifiés — points critiques

Référence complète : `docs/robot_action_state_processing_en.md` dans le repo. Résumé des points où les implémentations se trompent le plus souvent :

- **Action = 54D, état = 60D**, plus masques booléens de même taille.
- **Rotation-6D = les deux premières COLONNES** de la matrice de rotation, dans l'ordre `[R00, R10, R20, R01, R11, R21]`. Erreur classique : utiliser les lignes.
- **Actions de pose = `T_t⁻¹ · T_{t+k}`**, sorties en xyz + rotation vector. La translation relative est dans le repère **local courant**, pas une différence de positions monde.
- **L'état n'est jamais relativisé** — toujours l'observation absolue courante.
- **État effecteur : seul le xyz est normalisé**, la rotation-6D reste brute.
- **Slot état `[41:47]`** = direction de gravité corps + vitesse angulaire normalisée. Ce n'est **pas** une pose de base. Le slot action `[35:41]`, lui, est bien une pose relative de base.
- **Mains dextres → normalisation type gripper**, pas type action ordinaire.
- **Ordre d'exécution figé :** relativisation → normalisation → binarisation gripper (optionnelle) → rééchantillonnage → mapping vers les slots.
- **Actions : pas de zero-padding automatique.** Un module d'action plus court que son slot est une erreur de schéma. Les **états**, eux, sont zero-paddés en queue et le slot entier reste marqué valide.
- Conventions obligatoires en entrée : mètres, radians, quaternions `xyzw`, euler `xyz`. Le pipeline ne convertit rien.
- Repère base : x avant, y gauche, z haut, main droite. **Pas de repère miroir pour le bras droit.**

### Incohérence connue entre spec et implémentation

**[VÉRIFIÉ]** La doc du dataloader indique que l'échantillonnage est proportionnel au nombre d'échantillons de chaque source et que **l'implémentation actuelle n'applique pas la valeur `weight` configurée**.

Or la spec §12 fusionne les statistiques à **poids égaux par tâche**, en justifiant ça par l'hypothèse d'un échantillonnage équilibré par tâche. Le dataloader fait donc l'inverse de ce que la spec suppose.

**Conséquence pratique :** si les tâches G1-D ont des volumes très inégaux, les normalizers seront décalés par rapport à la distribution réellement vue. À garder en tête si des actions aberrantes apparaissent. Contournement : équilibrer les volumes par tâche, ou recalculer les statistiques en pondérant par le nombre d'échantillons (formule donnée en §12.8 de la spec).

---

## 7. Détails techniques vérifiés sur les modèles

**[VÉRIFIÉ]** `config.json` de UnifoLM-ER-Flow :
```json
"robot_state_dim": 120,
"robot_state_token": "<|robot_state|>",
"robot_state_token_id": 151678,
"robot_state_projector_hidden_size": null,
"hidden_size": 2560,
"architectures": ["Qwen3VLForConditionalGeneration"]
```

**[INFÉRÉ]** `robot_state_dim: 120 = 2 × 60`. L'hypothèse la plus naturelle est la concaténation `[état_60D | masque_60D]`, projetée par MLP vers 2560 et injectée à la position du token `<|robot_state|>`. **À confirmer en lisant le code du dataloader / du projector** — c'est une vérification rapide et elle conditionne tout le convertisseur.

**[VÉRIFIÉ]** Comparaison des checkpoints ER-1 vs ER-Flow : le seul ajout d'ER-Flow est `lm_head.weight` (délié, ~783 Mo en BF16 pour 152960×2560, ce qui explique intégralement l'écart de 8,89 → 9,69 Go) et `robot_state_projector.net.*`. Aucun codebook RVQ, aucun décodeur VQ-VAE dans ER-Flow. Les vocabulaires (153088 pour ER-1, 152960 pour ER-Flow) diffèrent uniquement par du padding d'alignement sur 128 — pas de tokens d'action ajoutés au vocabulaire texte.

---

## 8. Plan de travail proposé

### Phase 0 — Valider l'installation sans robot
1. Installer via `uv sync` (le projet utilise `uv`).
2. Télécharger `UnifoLM-WLA-1.0-Base` dans `playground/Pretrained_models/`.
3. Télécharger un dataset Dex1 (ex. `G1_Dex1_Stack_Block`), configurer `unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml`.
4. Lancer `eval_local_episode.py`. Il trace prédiction vs vérité terrain **en pose absolue et en représentation relative native** — c'est le meilleur moyen de comprendre le format avant d'y injecter des données G1-D.

### Phase 1 — Comprendre le format cible
5. Inspecter la structure d'un épisode Dex1 : clés, dimensions, dtypes.
6. Lire le dataloader pour confirmer la composition des 120D (hypothèse état+masque).
7. Lire `_ACTIVE_SLOT_SPECS` et le protocole du serveur pour cartographier exactement ce qui est supporté.

### Phase 2 — Convertisseur G1-D
8. Écrire le convertisseur état/action G1-D → 60D/54D + masques, piloté par config, conforme à la spec.
9. Tests aller-retour : normalisation → dénormalisation doit redonner les valeurs d'origine ; conversions de pose SE(3) idempotentes.
10. Valider le convertisseur en le faisant reproduire les tenseurs d'un épisode Dex1 officiel à partir de ses données brutes.

### Phase 3 — Déploiement
11. Intégrer `unitree_deploy` (repo `unifolm-world-model-action`) : déclarer une config `g1d_*` en s'inspirant de `g1_dex1`. Valider la chaîne caméras/DDS/effecteur avec `test_replay.py`, **sans modèle**.
12. Brancher le serveur d'action WLA. Patcher l'adapter si l'effecteur n'est pas Dex1 (§4).

### Phase 4 — Adaptation
13. Test zero-shot sur une tâche tabletop simple, jambes masquées. Noter le taux de succès comme baseline.
14. Collecter 50–200 démos téléopérées par tâche via `xr_teleoperate`.
15. Fine-tuner avec VLM gelé. Seuil de décision : si <50 % de succès après 200 démos, revoir la qualité des démos ou le recalage du repère base avant d'augmenter le volume.

### Compute
Le modèle fait 6B paramètres. L'inférence ne tournera pas sur le Jetson Orin NX embarqué. Prévoir un PC d'inférence relié au robot — l'architecture serveur/client du repo (websocket + msgpack) est faite pour ça.

---

## 9. Questions ouvertes à résoudre

| # | Question | Impact | Comment trancher |
|---|---|---|---|
| 1 | **Quel effecteur sur ce G1-D ?** | Détermine s'il faut patcher le serveur (§4) | Demander à l'utilisateur |
| 2 | Format des données de téléop existantes | Détermine le convertisseur | `ls` sur un épisode, `dataset.features` |
| 3 | Les 120D sont-ils bien état+masque ? | Fondation du convertisseur | Lire le dataloader |
| 4 | Caméra du rig de collecte WLA | Ampleur du recalage visuel | Config de preprocessing + métadonnées datasets |
| 5 | Où poser le repère base `B` du G1-D ? | Validité des normalizers | Comparer les URDF G1 / G1-D |
| 6 | G1-D Standard (base fixe) ou Ultimate (roulant) ? | Faut-il gérer `base_pose` ? | Demander à l'utilisateur |

---

## 10. Pièges à éviter

- **Ne pas fusionner les statistiques entre embodiments.** La spec §12.1 l'interdit explicitement, et c'est une source d'erreur silencieuse.
- **Ne pas se contenter d'entraîner la tête d'action.** La littérature montre que c'est insuffisant même en distribution. La recette officielle entraîne aussi le state projector — la suivre.
- **Ne pas supposer que le serveur supporte les mains dextres ou la base.** Il ne les supporte pas (§4).
- **Ne pas confondre le slot état `[41:47]`** (gravité + vitesse angulaire) **avec une pose de base**.
- **Ne pas zero-padder les actions** trop courtes : c'est une erreur de schéma à signaler, pas à corriger silencieusement.
- **Ne pas viser le one-shot in-context.** WLA n'a pas cette capacité ; c'est une revendication d'autres acteurs (GEN-1.5), non reproduite publiquement. Le régime réaliste est le few-shot par fine-tuning.
- **Ne pas sauter la Phase 0.** Valider le pipeline sur des données officielles avant d'y injecter des données G1-D évite de débugger deux problèmes à la fois.&&