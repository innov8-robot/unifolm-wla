# Constats — Portage de UnifoLM-WLA-1.0 sur Unitree G1-D

Ce fichier regroupe ce qu'on a vérifié dans le code pour notre cas d'usage. Il complète le briefing [Ma_Reflection.md](Ma_Reflection.md) et le corrige là où le code le contredit.

Mêmes niveaux de confiance que le briefing :

- **[VÉRIFIÉ]** — lu directement dans le code du dépôt.
- **[INFÉRÉ]** — déduction cohérente, à valider.
- **[INCONNU]** — question ouverte.

Sauf mention contraire, tous les constats viennent d'une **lecture statique du code**. Rien n'a encore été exécuté, ni sur données réelles ni sur le robot.

Dernière mise à jour : 29 septembre 2026, sur la base du commit `406faa3` d'Unitree, qui inclut la PR #14 sur le LoRA.

---

## 1. Notre matériel

**[VÉRIFIÉ, confirmé par l'utilisateur]**

| Élément | Notre G1-D |
|---|---|
| Effecteur | Dex1 (gripper deux doigts), version interne |
| Caméras | caméra binoculaire de tête et une caméra sur chaque poignet |
| Base | roulante, châssis différentiel (Flagship ou Ultimate) |

Cela tranche les questions 1 et 6 du briefing. Le Dex1 est l'effecteur nominal du serveur : **aucun patch n'est nécessaire pour les doigts**.

---

## 2. Correction importante : le robot-state projector est gelé

**[VÉRIFIÉ]** Le briefing affirme, dans ses sections 3 et 10, que la recette officielle entraîne la tête DiT **et** le robot-state projector. C'est faux.

- Le projecteur est attaché **à l'intérieur** de l'interface VLM, à l'emplacement `qwen_vl_interface.model.robot_state_projector`, dans [QWen3.py](../unifolm_wla/model/modules/vlm/QWen3.py#L125-L141).
- La recette gèle `qwen_vl_interface` en entier, dans [mmdit_finetune_frozen_vlm.yaml](../unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml).
- La fonction de gel parcourt tous les paramètres du module et les met à `requires_grad = False`, dans [trainer_tools.py](../unifolm_wla/training/trainer_utils/trainer_tools.py#L188-L231). Le gel s'applique après l'attachement du projecteur.
- **Seule la tête DiT s'entraîne.** Le commentaire du YAML, qui dit le contraire, est erroné.

**Contournement [INFÉRÉ, non testé]** : geler le VLM par sous-modules au lieu de le geler en bloc.

```yaml
trainer:
  freeze_modules: "qwen_vl_interface.model.model,qwen_vl_interface.model.lm_head"
```

**[VÉRIFIÉ le 29 septembre 2026]** Le contournement fonctionne. Comptage module par module, modèle construit puis gelé :

| Réglage | VLM entraînable | Projecteur entraînable | Tête DiT entraînable |
|---|---|---|---|
| `qwen_vl_interface`, recette officielle | 0 | **0** sur 6,87 M | 1 390,48 M |
| `qwen_vl_interface.model.model,qwen_vl_interface.model.lm_head` | 0 | **6,87 M** | 1 390,48 M |

Un vrai lancement confirme 1 397,3 M paramètres entraînables. La recette G1-D l'applique : `unifolm_wla/config/training/g1d_finetune_frozen_vlm.yaml`.

**[VÉRIFIÉ]** Ni la recette officielle ni la nôtre ne tiennent sur une RTX 5090 Laptop, qui offre 23,4 Go utilisables. Les deux saturent au premier passage arrière. Avec l'optimiseur déporté en mémoire vive, la nôtre tourne à environ 1,7 s par pas. Le commentaire officiel parle pourtant d'« un seul GPU de 24 Go ».

**À corriger avant tout fine-tuning G1-D.**

### Le fine-tuning LoRA ne change rien

**[VÉRIFIÉ]** Unitree a publié le fine-tuning LoRA après le briefing, dans la PR #14. Le briefing le disait non publié.

- La config [mmdit_lora_frozen_vlm.yaml](../unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml) gèle toujours `qwen_vl_interface` en entier.
- Les adaptateurs LoRA ne visent que les couches d'attention de la tête DiT.
- Le gel épargne désormais les paramètres dont le nom contient `lora_`. Le projecteur n'en a aucun, donc **il reste gelé en LoRA aussi**.

**[INFÉRÉ]** En LoRA, la bibliothèque peft gèle aussi les poids de base de la tête DiT. Seuls les adaptateurs s'entraîneraient alors. À confirmer avec `print_trainable_parameters`.

---

## 3. Entrée du projecteur : 120D = état ⊕ masque

**[VÉRIFIÉ]** Tranche la question 3 du briefing. Code : [QwenMMDiT.py](../unifolm_wla/model/framework/VLM4A/QwenMMDiT.py#L68-L105).

- L'entrée est la concaténation de l'**état unifié 60D** et de son **masque 60D**.
- Le projecteur est un MLP à deux couches, Linear, SiLU puis Linear, de 120 vers 2560.
- Sa sortie remplace l'embedding du token `<|robot_state|>` dans le prompt du VLM.
- **En fine-tuning, l'état n'entre que par ce token.** La tête DiT ne reçoit pas d'état, car `action_model.state_dim` vaut 0 dans la config.

Augmentation pendant l'entraînement, tirée une fois par exemple :

| Probabilité | Traitement |
|---|---|
| 10 % | état et masque mis à zéro |
| 30 % | bruit gaussien d'écart-type 0,1 sur les dimensions valides |
| 60 % | inchangé |

**[INFÉRÉ]** Le modèle a donc déjà appris à fonctionner avec un état absent ou bruité. C'est plutôt favorable pour le zero-shot sur G1-D.

---

## 4. Actions des bras : relatives au début du chunk

**[VÉRIFIÉ]** Code : [se3_utils.py](../unifolm_wla/dataloader/multi_source_dataset/se3_utils.py#L162-L180) et [single_source_dataset.py](../unifolm_wla/dataloader/multi_source_dataset/single_source_dataset.py#L569-L611).

### Côté dataset

Pour chaque bras :

1. **L'ancre** est la pose effecteur **mesurée** à l'instant t, prise dans l'état.
2. **Les cibles** sont les 30 poses effecteur **commandées** de t à t+29, soit une seconde à 30 images par seconde.
3. Chaque cible est exprimée dans le repère de l'ancre : `T_rel_k = T_ancre⁻¹ · T_cmd(t+k)`.
4. Le résultat est converti en xyz et vecteur de rotation, soit 6 valeurs, puis normalisé en z-score avec des statistiques relatives dédiées, stockées dans `relative_stats.json`.

Ce ne sont **pas des deltas entre deux pas successifs**. Chaque pose du chunk est relative à la même ancre.

Le chunk commence à t, donc **la première valeur n'est pas nulle**. Elle représente l'écart de suivi entre la commande et la mesure au même instant.

### Côté modèle et serveur

- Le modèle sort un chunk de 30 pas de 54 valeurs normalisées. Pour les bras, ces valeurs sont relatives.
- Le serveur dénormalise, puis compose chaque pose avec l'ancre mesurée reçue dans l'observation : `T_abs = T_ancre · T_rel`. Code : [action_server_wbc_msgpack_unitree.py](../model_server/action_server_wbc_msgpack_unitree.py#L131-L142).
- Le client reçoit donc des **poses absolues** dans le repère base, au format xyz + roulis-tangage-lacet, en euler `xyz`.

### Ce qui reste en valeurs absolues

Ces canaux sont seulement normalisés, jamais relativisés :

- les grippers ;
- la taille et les jambes, en angles articulaires ;
- la commande de base, en vitesses et en hauteur ;
- l'état envoyé au modèle, qui est toujours l'observation absolue courante.

### Conséquences pour nos enregistrements G1-D

- **Enregistrer deux poses distinctes par bras** : la pose mesurée par cinématique directe dans l'état, et la pose commandée par la téléop dans l'action, toutes deux dans le même repère base. Si on enregistre la même valeur des deux côtés, le premier pas relatif vaut toujours zéro, ce qui n'est pas la distribution vue par le modèle.
- **Soigner l'ancre au déploiement.** Une erreur sur la pose mesurée décale toute la trajectoire absolue du chunk, même si la prédiction relative est parfaite.

**À vérifier sur données réelles** : sur un épisode Dex1 officiel, comparer la pose mesurée et la pose commandée au même instant. On doit voir un petit écart non nul.

---

## 5. Base roulante : la commande de base existe déjà

**[VÉRIFIÉ]** Le briefing, section 4, dit que le serveur ne gère pas la base. C'est vrai pour la **pose relative de base**, qui n'existe que dans le dataset whole-body. En revanche, le serveur produit déjà une **commande de base** en vitesses, sans aucun patch.

- La liste des slots actifs contient `base_command`, composé de `base_vx_vy`, `base_vw` et `height`, dans [unifolm_wla_action_adapter.py](../model_server/unifolm_wla_action_adapter.py#L18-L27).
- Le serveur la renvoie au client sous `action.base_command`, en 4 valeurs par pas, et sous `action.pivot`, en 7 valeurs par pas qui ajoutent les angles de taille.
- Sur le G1 bipède, c'est la consigne envoyée au contrôleur de marche. Le dataset Dex1 la remplit.

Plages vues à l'entraînement, tirées de `stats/stats.json` :

| Valeur | Quantile 1 % | Médiane | Quantile 99 % |
|---|---|---|---|
| vx, vitesse avant | −0,40 m/s | 0 | 0,67 m/s |
| vy, vitesse latérale | −0,48 m/s | 0 | 0,48 m/s |
| vw, vitesse de rotation | −1,43 rad/s | 0 | 1,49 rad/s |
| height, hauteur du bassin | 0,38 m | 0,74 m | 0,79 m |

Les minimums et maximums bruts contiennent des valeurs aberrantes, jusqu'à ±119. La normalisation utilise les quantiles, donc ces valeurs sont sans effet sur le modèle.

### Adaptations côté client G1-D [INFÉRÉ]

- **vy** : un châssis différentiel ne peut pas glisser de côté. Ignorer la valeur prédite et enregistrer vy = 0 dans nos démos.
- **height** : c'est la hauteur du bassin du G1 debout ou accroupi, pas une position de colonne. Définir une correspondance fixe entre les deux, telle qu'une hauteur de 0,74 m donne la même hauteur de caméra qu'un G1 debout. À calculer depuis les deux URDF.
- **vx** : notre châssis atteint 1,5 m/s, mais le modèle n'a jamais vu plus de 0,67 m/s. Brider la vitesse au début.
- **Taille** : ~~masquer ce slot~~ **corrigé, voir la section 9** : envoyer [0, 0, tangage du buste G1-D], slot valide. Ignorer les prédictions de lacet et de roulis.
- **Jambes** : ~~masquer les slots jambes~~ **corrigé, voir la section 9** : envoyer la posture debout du G1, slots valides. Ignorer leurs prédictions.

---

## 6. Caméras attendues

**[VÉRIFIÉ]** Le serveur et la config de données attendent trois images, sous trois rôles :

| Rôle | Clé dans le dataset Dex1 |
|---|---|
| `head_left` | `observation.images.head_stereo_left_rec`, vue gauche **rectifiée** de la caméra stéréo de tête |
| `cam_wrist_left` | `observation.images.wrist_left` |
| `cam_wrist_right` | `observation.images.wrist_right` |

Les images sont redimensionnées en 336×448 (hauteur × largeur). Source : [unitree.yaml](../unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml).

**[INFÉRÉ]** La tête est bien une caméra stéréo, ce qui correspond à la binoculaire du G1-D. Il faudra fournir la vue gauche **rectifiée**.

**[INCONNU]** Résolution native, champ de vision et intrinsèques des caméras de collecte. À lire dans les métadonnées d'un épisode Dex1 téléchargé, puis à comparer à nos caméras.

---

## 7. Anomalies trouvées dans le dépôt

**[VÉRIFIÉ]**

- **Mauvais chemin de dataset** : dans `unitree.yaml`, l'entrée `UnifoLM_WBT_Dataset` a pour chemin `UnifoLM_G1_Dex1_Dataset/`. Avec les deux entrées actives, le Dex1 est chargé deux fois et le WBT jamais. Pour nous, garder seulement l'entrée Dex1.
- **Chemin par défaut cassé** : `train_unifolm_wla.py` et `QWen3.py` ont une valeur par défaut `--config_yaml` qui pointe vers un fichier SimplerEnv absent. Toujours passer la config explicitement.
- **Options sans effet** : le script d'entraînement from scratch passe des options `omnivggt_fusion`, `action_tokenizer`, `rtc` et `max_delay`. Aucun code ne les lit.
- **Docstring fausse** : celle de `eval_local_episode.py` cite un chemin `examples/pretrain/...` qui n'existe pas. Le bon chemin est `examples/unifolm_wla/eval_files/unitree/eval_local_episode.py`.
- **Config de debug supprimée** : `unitree_debug.yaml` a été retiré par Unitree dans la PR #14.
- **Pondération des sources** : le dataloader échantillonne proportionnellement au nombre d'échantillons et ignore le champ `weight`. C'est déjà noté dans le briefing, section 6.

---

## 7 bis. Sim MuJoCo du G1-D : état et écarts avec WLA

**[VÉRIFIÉ]** Le smoke test passe entièrement, lancé le 29 septembre 2026 dans l'environnement conda `unitree_lerobot` avec MuJoCo 3.10 et Pinocchio 3.9 :

```
MUJOCO_GL=egl python sim/smoke.py    # depuis la racine du dépôt
15/15 vérifications passées
```

Tenue sous gravité, IK aller-retour, pose de travail, suivi cartésien à 30 Hz, refus d'une cible hors d'atteinte, pinces, sauvegarde et restauration d'état et rendu des trois caméras fonctionnent.

### Caméra de tête calée sur l'entraînement (29 septembre 2026)

L'utilisateur a confirmé que le G1-D utilise la **caméra stéréo de tête stock**. La caméra RGBD du torse a été retirée de la sim, qui ne rend plus que du RGB.

**[VÉRIFIÉ]** Métadonnées des datasets G1 Dex1 en format LeRobot v3.0, par exemple `G1_Dex1_Stack_Block` :

- toutes les vues sont en **640×480 à 30 fps** ;
- la tête existe en brut (`head_stereo_left`) et en rectifié (`head_stereo_left_rec`). WLA utilise la vue rectifiée ;
- les datasets enregistrent la pose d'une caméra `d435` et celle du buste dans le repère base.

**[VÉRIFIÉ]** La pose `d435` dans `torso_link` est **constante** sur tout un épisode : xyz (0.0576, 0.0175, 0.4299) et tangage 0.8308 rad. C'est un montage fixe.

**[VÉRIFIÉ]** Le G1 manipule le **buste penché** par rapport au bassin. Médiane globale des statistiques d'entraînement : 0.166 rad. Selon la tâche : 0.127 rad pour Stack_Block et 0.182 rad pour Wipe_Table. Buste droit, la caméra plongerait environ 10° de moins.

**[INFÉRÉ]** Intrinsèques de l'œil gauche rectifié, estimées en projetant les poses effecteur enregistrées sur 7 images :

| Paramètre | Estimation |
|---|---|
| Focale | environ 310 px à 640×480 |
| Champ | environ 91° horizontal et 75° vertical |
| Décalage de l'œil gauche | environ 7 cm vers la gauche de la pose `d435` |
| Erreur de reprojection | 12 à 18 px |

Une hypothèse plus physique, avec l'œil à 3 cm de l'axe, suit aussi bien les pinces à 10 ou 20 px près. Le décalage latéral est donc peu contraint. **À remplacer par la vraie calibration de la stéréo de tête de notre G1-D.**

Ce qui a été fait dans la sim :

- caméra `head_left_cam` sur `torso_link`, à la pose ci-dessus, fovy 75° ;
- noms des vues alignés sur les rôles WLA : `head_left`, `cam_wrist_left` et `cam_wrist_right` ;
- buste incliné à 0.166 rad par `go_ready`, avant de sortir les bras ;
- caméra abaissée de 1 cm : la tête du G1-D est montée 1 cm plus bas sur `torso_link` que celle du G1 ;
- buste compensé en gravité : la FK colle à MuJoCo à 0,1 mm près, buste droit ou penché.

**Conséquence pour le vrai robot [INFÉRÉ]** : pour coller à l'entraînement, le G1-D devrait lui aussi manipuler buste penché d'environ 0.166 rad, avec son joint de tangage.

**Encore différent des images réelles** : la scène elle-même. La table est petite, le sol uniforme, l'éclairage simple, et les intrinsèques des caméras de poignet ne sont pas calées.

### Écarts avec WLA : état au 29 septembre 2026

| Point | État dans la sim |
|---|---|
| Repère base | **Fait** : bassin virtuel du G1, voir la section 9 |
| Effecteur | **Fait** : `ee_pose_wla` et `track_ee_wla`, voir la section 9 |
| Caméra haute | **Fait**, intrinsèques estimées |
| Pose de départ | **Fait** : angles médians de départ du G1 |
| Hauteur de table | **Fait** : table remontée de 13 cm, estimation à ±3 cm |
| Caméras poignet | **[INCONNU]** fovy 110° non calé sur les données |
| Pince | unité et sens connus, correspondance avec la sim non calibrée |
| Base roulante | aucun actionneur de roue, non testable en sim |
| Bord de table | plus loin du robot que dans les données |
| Rendu | matériaux, éclairage et décor différents des images réelles |

### Prochaine étape proposée

Écrire un client sim qui joue le rôle du robot pour le serveur WLA :

1. rendre les trois vues et lire les poses effecteur et les pinces ;
2. les exprimer dans `B` et les envoyer au serveur ;
3. appliquer le chunk reçu avec `track_tcp` et `set_gripper`, pas à pas à 30 Hz.

Cela permet un test zero-shot en boucle fermée sans robot, dès que les poids sont téléchargés.

---

## 8. État des questions ouvertes du briefing

| # | Question | État |
|---|---|---|
| 1 | Effecteur | **Résolu** : Dex1 interne. |
| 2 | Format de nos données de téléop | Ouvert. |
| 3 | 120D = état ⊕ masque ? | **Résolu** : oui, voir la section 3. |
| 4 | Caméra du rig de collecte | **Résolu** : tête stéréo, vue gauche rectifiée, et deux poignets. Intrinsèques estimées. |
| 5 | Repère base du G1-D | **Résolu** : bassin virtuel, voir la section 9. |
| 6 | Base fixe ou roulante | **Résolu** : roulante, pilotable par la commande de base, voir la section 5. |

Nouvelles questions :

| # | Question | Comment trancher |
|---|---|---|
| 7 | Le gel par sous-modules libère-t-il bien le projecteur seul ? | **Résolu** : oui, voir la section 2. |
| 8 | Correspondance entre la hauteur G1 et la position de la colonne G1-D | Comparer les URDF à hauteur de caméra égale. |
| 9 | Écart commande/mesure dans les données officielles | **Résolu** : décalage constant d'environ 1 cm, voir la section 9. |
| 10 | Point et axes de l'effecteur G1 dans les datasets | **Résolu** : voir la section 9. |
| 11 | Sens et unité de la pince Dex1 dans les datasets | **Résolu pour le sens** : voir la section 9. Correspondance physique à calibrer. |
| 12 | Caméra haute réelle du G1-D | **Résolu** : stéréo de tête stock. |
| 13 | Intrinsèques et baseline de la stéréo de tête | Calibrer celle de notre G1-D. |
| 14 | Le G1-D peut-il manipuler buste penché d'environ 0.166 rad ? | Tester sur le robot. |
| 15 | Hauteur exacte de la table dans les datasets G1 | Reconstruire le plan de table avec la caméra calibrée. |
| 16 | Intrinsèques des caméras de poignet | Même méthode que la tête, sur les vues de poignet. |

---

## 9. Contrat iso WLA pour le G1-D

Ce que le client G1-D, réel ou sim, doit envoyer au serveur WLA pour rester dans la distribution d'entraînement. Tout est **[VÉRIFIÉ]** dans le code, les datasets G1 Dex1 ou l'URDF officiel du G1, sauf mention contraire.

### Images

- **Trois images seulement**, dans cet ordre : `head_left`, `cam_wrist_left`, `cam_wrist_right`. Aucun dataset ne charge l'œil droit : les configs de données ne listent que l'œil gauche pour la tête.
- **Tête** : œil **gauche rectifié**, `head_stereo_left_rec`, pour les datasets Dex1. Le dataset whole-body utilise l'œil gauche **brut**.
- **Format source** : 640×480 à 30 fps.
- **Taille vue par le modèle** : 320×448. À l'entraînement, l'image passe en 336×448 avec antialias, puis le processeur Qwen la ramène à 320×448. Le serveur redimensionne directement en 320×448, sans antialias. La grille de patches est la même, mais le rééchantillonnage diffère un peu.
- **Couleurs** : le serveur attend du **BGR** et le convertit en RGB. La sim rend du RGB : il faut convertir avant l'envoi.
- **Limite de la vérification** : la config publiée du modèle Base ne liste pas ses datasets. Ce qui précède vient de la config de données du dépôt, du serveur officiel et des métadonnées des datasets publics. Voir « Données d'entraînement » ci-dessous.

### Données d'entraînement du modèle Base

**[VÉRIFIÉ]** Le dépôt Hugging Face `unitreerobotics/UnifoLM-WLA-1.0-Base` ne contient **pas** de données. Il contient les poids, une config dont la liste de datasets est vide, les statistiques de normalisation et le tokenizer. La collection officielle UnifoLM-WLA-1.0 le relie à deux collections de datasets, qui correspondent aux deux clés de normalisation.

| Collection | Datasets | Frames avec pose effecteur | Heures |
|---|---|---|---|
| UnifoLM_G1_Dex1_Dataset | 72 | 11,7 M | 108 |
| UnifoLM_WBT_Dataset | 58 | 11,3 M | 105 |
| UniBot-V1 Challenge Dataset, Dex1 | 32 | 18,3 M | 169 |
| **Total public** | 162 | 41,3 M | 382 |
| Statistiques du modèle Base | — | 54,0 M | 500 |

- Les normaliseurs du Base sont **identiques** aux statistiques du dépôt, par exemple les quantiles de la pose effecteur et de la pince.
- Les datasets publics couvrent **76 %** des frames des statistiques. **[INFÉRÉ]** Le reste n'est pas publié.
- **Caméras de tête des 72 datasets Dex1** :
  - 56 ont les deux yeux, bruts et rectifiés ;
  - 5 n'ont que les yeux bruts. Avec la config du dépôt, leur vue de tête manque et le rôle `head_left` disparaît du prompt ;
  - 11 sont dans l'ancien format, avec des vues nommées `cam_left_high` et sans pose effecteur. Ils sont inutilisables tels quels par la config WLA.
- **Whole-body** : 54 datasets sur 55 lisibles n'ont que les yeux bruts, ce qui est cohérent avec la config qui y prend `head_stereo_left`.
- **UniBot** : les 32 datasets n'ont que les yeux **bruts**. Ils ne sont pas dans la config du dépôt.

**[VÉRIFIÉ]** Répartition des frames publiques avec pose effecteur selon la vue de tête :

| Vue de tête disponible | Frames | Part |
|---|---|---|
| Œil gauche brut seulement : UniBot, WBT, 5 Dex1 | 30,8 M | 75 % |
| Œil gauche rectifié et brut : 56 Dex1, 1 WBT | 10,4 M | 25 % |

**Conséquence** : la vue rectifiée n'est **pas** majoritaire. La config Dex1 du dépôt prend la rectifiée, mais la majorité des données publiques n'a que la brute. **[INCONNU]** Quelle vue Unitree a donnée au modèle Base pour UniBot et pour les données privées.

**[VÉRIFIÉ]** Écart entre les deux vues, mesuré sur Stack_Block par appariement de points :

| Mesure | Valeur |
|---|---|
| Zoom du rectifié par rapport au brut | ×1,116 |
| Décalage de l'axe optique | 3,3° |
| Déplacement médian d'un point | 24 px |
| Résidu après homographie, distorsion | 1 à 2 px |
| Champ vertical, rectifié puis brut | environ 75° puis 81° |

La sim propose les deux vues : `G1DSim(head_view="rec")` par défaut, ou `head_view="raw"`. **Recommandation** : ne pas trancher à l'aveugle. Tester les deux en zero-shot, puis fine-tuner avec la vue que l'on enregistre sur le G1-D. Le modèle a vu les deux.
- Toutes les vues publiques sont en 640×480 à 30 fps.
- **Œil droit** : disponible dans presque tous les datasets, mais aucune config ne le charge.

### Prompt

Le serveur construit exactement le prompt d'entraînement. Les noms de rôle y sont écrits en toutes lettres :

```
head_left: <image>
cam_wrist_left: <image>
cam_wrist_right: <image>
Task: <instruction>
State: <|robot_state_implicit_stats|><|robot_state|>
Control Mode: Arms: EE, LOW BODY: JOINT
```

La ligne « LOW BODY » vient du type de bras `dual_with_legs` du Dex1. Le serveur la met toujours, ce qui est iso.

### Repère base

- Chez le G1, la base est le **bassin**. Le torse s'y trouve à une translation fixe de (-0.004, 0, 0.044) m, puis tourné du tangage de la taille.
- Sur le G1-D, la base est un **bassin virtuel** : le repère placé sous le torse avec cette même relation, en utilisant le tangage mesuré du buste. Il reste horizontal.
- **Validation** : les angles de départ du G1, appliqués au G1-D dans ce repère, redonnent la pose effecteur du G1 à environ 5 mm près.

### Effecteur

- Chez le G1, `*_ee_pose_gripper_base` est **exactement** `*_wrist_yaw_link` décalé de 0.105 m le long de son axe x, sans rotation. La FK sur un épisode donne un écart nul.
- Sur le G1-D, on garde le même point par rapport à la pince : `*_wrist_yaw_link` + (0.105, ±0.003, 0), avec l'orientation du poignet.
- Le TCP historique de la sim est 4 cm plus loin et tourné de 90°. Il ne faut **pas** l'envoyer au modèle.
- **Format** : euler `xyz` extrinsèque dans les datasets. Le serveur attend xyz + rot6d, soit les deux premières colonnes de la matrice.
- **Ordre des joints du bras dans les datasets** : ordre standard du G1, avec le poignet en roulis, tangage, lacet. Les noms des métadonnées, qui disent lacet, roulis, tangage, sont **faux**.

### Bras et buste

- Les bras G1 et G1-D sont identiques, sauf le dernier lien du poignet, plus court de 5 mm sur le G1-D.
- **Buste** : tangage d'environ 0.166 rad en médiane globale, entre 0.13 et 0.18 rad selon la tâche.
- **Pose de départ** : angles médians de départ du G1, repris dans `G1_START_Q` de la sim.

### État hors bras

- **Jambes** : slots **valides** à l'entraînement Dex1, avec le G1 debout. Il faut envoyer la posture debout du G1, par exemple hanche en tangage -0.41 rad et genou 0.65 rad. Le détail est dans `G1_STANDING_LEGS` de la sim. Le briefing supposait ces slots masqués : c'est faux pour les données Dex1.
- **Taille** : slot valide. Envoyer [lacet 0, roulis 0, tangage du buste G1-D]. Chez le G1, le tangage de la taille est égal au tangage du buste.
- **Pince** : unité Dex1 brute. La valeur **monte à l'ouverture** : environ 4,5 ouverte au repos, environ 2,4 fermée sur un cube de 4 cm, plage totale de 0 à 5,5. La correspondance avec la position des doigts de la sim reste à calibrer.
- **Clé de normalisation** : `UnifoLM_G1_Dex1`.

### Actions

- Bras relatifs à l'ancre mesurée, chunk de 30 pas à 30 Hz, voir la section 4.
- **Commande et mesure** : décalage constant d'environ 1 cm en médiane, sans retard. C'est l'erreur de suivi du G1.
- **Commande de base** dans les données Dex1 de table : vitesses nulles, hauteur de bassin environ 0,73 m.

### Scène

- **Table** **[INFÉRÉ]** : environ 0,07 m au-dessus du bassin G1, à ±3 cm, estimé depuis la hauteur des prises. Le bassin G1 est à environ 0,73 m du sol.
- **G1-D réel** : avec la colonne en butée basse, son bassin virtuel est à environ 0,80 m. Pour être iso, sa table devrait être à environ 0,87 m, ou plus haut si la colonne monte.


---

## 10. Enregistrer un dataset G1-D pour WLA

Constats du 29 septembre 2026, tirés du code de `xr_teleoperate` et de `unitree_lerobot`, branches principales.

### Ce que fait xr_teleoperate

- **[VÉRIFIÉ]** C'est l'outil de téléopération et d'enregistrement d'Unitree, avec un casque XR comme le Vision Pro ou le PICO 4.
- **[VÉRIFIÉ]** Il connaît les Dex1 câblées en interne : `--ee dex1_internal`, qui pilote les pinces par les moteurs 31 et 33 de la commande bas niveau du G1.
- **[VÉRIFIÉ]** `dex1_internal` exige `--arm G1_29` et interdit `--motion`.
- **[VÉRIFIÉ]** Aucun `--arm` ne correspond au G1-D. Le code ne mentionne ni roues, ni colonne, ni G1-D.
- **[INCONNU]** Le G1-D accepte-t-il la commande bas niveau du G1 29 DoF, avec 35 moteurs sur `rt/lowcmd` ? **À vérifier avant tout essai**, bras dégagés et gains faibles. Une mauvaise correspondance des indices moteurs commanderait les mauvais joints.

### Ce qu'il enregistre

Pour chaque pas, à 30 Hz par défaut :

- **Images** : en JPEG. La tête binoculaire est coupée en deux moitiés **brutes**, gauche et droite. Les poignets sont ajoutés s'ils sont activés dans teleimager.
- **État** : angles **mesurés** des deux bras et position des pinces.
- **Action** : angles **commandés** des bras, sortie de l'IK, et commande des pinces.

Il n'enregistre **pas** les poses effecteur, la pose du buste, la pose caméra, les jambes ni la commande de base.

### Ce que produit unitree_lerobot

- **[VÉRIFIÉ]** Le convertisseur public JSON → LeRobot produit l'**ancien format** : un vecteur `observation.state` unique et des vues `cam_left_high`. C'est le format des 11 datasets Dex1 inutilisables par la config WLA.
- **[INFÉRÉ]** Le format WLA vient d'un pipeline Unitree non publié, qui recalcule les poses par cinématique directe. Le champ `recomputed_ee_valid` va dans ce sens, tout comme l'écart constant d'environ 1 cm entre commande et mesure, qui correspond à FK(angles commandés) comparé à FK(angles mesurés).

### Chaîne proposée

1. **Images** : teleimager, avec la tête binoculaire et les deux poignets, à 640×480 par vue.
2. **Téléopération** : si le G1-D est compatible, lancer :
   ```
   python teleop_hand_and_arm.py --arm=G1_29 --ee=dex1_internal --record --frequency 30 --task-name <nom> --task-goal "<instruction>"
   ```
3. **Conversion** : un convertisseur maison, JSON → LeRobot v3 au format WLA, **à écrire** :
   - effecteur état = FK des angles mesurés, effecteur action = FK des angles commandés, dans la base WLA et avec l'effecteur WLA de la section 9 ;
   - euler `xyz` pour l'enregistrement ;
   - jambes G1 debout, taille à [0, 0, tangage du buste], commande de base, poses buste et d435 ;
   - noms de colonnes des datasets G1 Dex1 v3 ;
   - vue de tête brute sous `head_stereo_left`, et rectifiée seulement si on calibre la stéréo.
4. **Normalisation** : garder les statistiques précollectées du dépôt, clé `UnifoLM_G1_Dex1`. Ne pas les recalculer ni les fusionner.

### Points iso propres au vrai G1-D

- **[INCONNU]** Format de la stéréo de tête du G1-D. La fiche produit annonce 3840×1200, soit 1920×1200 par œil, en 16:10, et 115° de champ. Les données G1 sont en 640×480 par œil, en 4:3, avec environ 91° horizontal en rectifié. Il faudra probablement **recadrer au centre** au format 4:3 et au champ du G1, puis réduire en 640×480. À décider après calibration.
- **[INFÉRÉ]** Table à environ 0,87 m avec la colonne en butée basse, et buste penché d'environ 0.166 rad, voir la section 9.

---

## 11. Premier test zero-shot en sim (29 septembre 2026)

**[VÉRIFIÉ]** La boucle fermée sim ↔ serveur WLA fonctionne avec le modèle Base :

- **Inférence** : environ 0,4 s par requête, sur une RTX 5090 Laptop, sans flash-attention.
- **Réception** : le serveur reçoit les trois images dans le bon sens et avec les bonnes couleurs, d'après son image de debug.
- **Exécution** : aucune cible IK refusée sur 10 chunks de 20 pas.

**[VÉRIFIÉ]** Un essai par vue de tête, instruction « pick up the black part and put it in the box » :

| Vue de tête | Comportement |
|---|---|
| Rectifiée | trajectoires presque immobiles, main droite qui remonte d'environ 6 cm, pince ouverte |
| Brute | main droite rapprochée de 4 cm vers la pièce, puis pince **fermée**, mais environ 12 cm trop haut |

**[INFÉRÉ]** Le modèle réagit plus à la vue brute, majoritaire dans les données publiques. **Non concluant** : un seul essai par vue, avec un échantillonnage aléatoire.

Écarts qui expliquent probablement l'échec :

- **Scène** : rendu de sim, pièce inconnue, décor différent.
- **Caméras de poignet** : pas calées.
- **Pince** : correspondance supposée linéaire avec l'unité Dex1.
- **Buste** : réglé à 0.166 rad dans la sim, alors que `mon_test` montre environ 0,09 rad sur le robot, si l'indice moteur est le bon.

Le fine-tuning sur nos démos reste la voie attendue. Voir la section 5 du briefing.

---

## 12. Validation de toute la chaîne en sim (29 septembre 2026)

**[VÉRIFIÉ]** Tâche : saisir un cube rouge de 4 cm, tiré au hasard dans une zone de 10 × 14 cm devant le bras droit, puis le soulever d'au moins 5 cm.

1. **Démos** : 150 démos d'un expert scripté, gardées sur 153 essais. L'expert garde l'orientation de pince de départ du G1. Elles sont enregistrées au format xr_teleoperate, avec la vue de tête brute.
2. **Conversion** : par `g1d_wla.convert_teleop`, le même convertisseur que pour le robot. Le dataset fait 23 700 frames.
3. **Fine-tuning** : 3 000 pas, lots de 2, VLM gelé, projecteur et tête DiT entraînés, en 1 h 26 sur la RTX 5090 Laptop.
4. **Évaluation** : en boucle fermée, sur 30 positions jamais vues.

| Modèle | Pas par épisode | Réussites |
|---|---|---|
| Base, zero-shot | 200 | 0 / 20 |
| Fine-tuné | 200 | 4 / 30 |
| Fine-tuné | 300 | **23 / 30** |

**Ce que ça valide** :

- les repères WLA, avec la base en bassin virtuel et l'effecteur WLA ;
- l'unité de pince ;
- les clés du convertisseur et la config de données ;
- les normaliseurs du Base ;
- le correctif du projecteur ;
- l'inférence en boucle fermée.

Le modèle apprend une nouvelle tâche à partir de 150 démos au format que produira le robot.

**Ce que ça ne valide pas** : le passage au réel. Le rendu de sim est loin des vraies images, et l'expert est parfaitement régulier.

**[CORRIGÉ, voir la section 13]** La lenteur venait surtout de l'exécution partielle des chunks, pas du modèle.

---

## 13. Vitesse d'exécution et nombre de démos (29 septembre 2026)

Même tâche cube et mêmes 30 positions d'évaluation que la section 12. Nouvelles mesures :

- **Temps de réussite** : le pas où le cube dépasse 5 cm de levée pour la première fois. L'expert scripté réussit en **127 pas** en médiane.
- **Critère de réussite** : le cube a été levé de plus de 5 cm **et** l'est encore à la fin.

### La lenteur venait de l'exécution partielle des chunks

**[VÉRIFIÉ]** Hors boucle, sur 60 extraits du dataset, le modèle à 150 démos prédit des mouvements de la **bonne amplitude** : le rapport entre prédiction et expert à t+29 vaut 1,02 à 1,04, avec une erreur de position médiane de 0,4 cm. Le modèle n'est donc pas lent en soi.

**[VÉRIFIÉ]** En boucle fermée, c'est le nombre de pas exécutés par chunk qui compte :

| Pas exécutés par chunk | Réussites | Pas médian jusqu'à la réussite |
|---|---|---|
| 30, chunk entier | **28 / 30** | 151 |
| 20 | 24 / 30 | 222 |
| 10 | 8 / 30 | 266 |

**[INFÉRÉ]** Chaque nouveau chunk repart de la pose mesurée, qui traîne légèrement derrière la commande à cause du suivi des servos. Chaque chunk recommence aussi une accélération depuis l'arrêt, comme l'expert au début de chaque phase. Plus on replanifie souvent, plus on perd de temps.

### Real-time chunking par inpainting

**[VÉRIFIÉ]** Nous avons ajouté un préfixe optionnel à l'échantillonnage flow matching de la tête d'action. On l'impose par inpainting à chaque pas d'intégration. Aucun réentraînement n'est nécessaire.

- **Préfixe** : le serveur garde le chunk précédent en poses absolues, et ré-exprime sa suite dans le repère de la nouvelle ancre mesurée.
- **Réinitialisation** : `policy_reset` vide cette mémoire.
- **Comportement par défaut** : sans préfixe, rien ne change.
- **Code** : `MMDiT_ActionHeader.predict_action`, `ActionServerWBCMsgpack._rtc_prefix`, et l'option `--rtc-prefix` de `sim/wla_client.py`.

| Modèle | Réglage | Réussites | Pas médian |
|---|---|---|---|
| 150 démos | 10 pas, sans préfixe | 8 / 30 | 266 |
| 150 démos | 20 pas + préfixe de 10 | 20 / 30 | 172 |
| 150 démos | 15 pas + préfixe de 15 | 24 / 30 | 160 |
| 150 démos | 10 pas + préfixe de 20 | 23 / 30 | **148** |

Le préfixe rend la replanification fréquente utilisable : à 10 pas par chunk, on passe de 8 à 23 réussites. C'est le réglage le plus rapide, mais il n'égale pas la fiabilité des chunks entiers.

### Nombre de démos

**[VÉRIFIÉ]** Même recette, 3 000 pas d'entraînement, sur les 25 premières démos seulement :

| Démos | Réglage | Réussites | Pas médian |
|---|---|---|---|
| 150 | chunk entier | 28 / 30 | 151 |
| **25** | chunk entier | **30 / 30** | 173 |
| 150 | 10 pas + préfixe de 20 | 23 / 30 | 148 |
| 25 | 10 pas + préfixe de 20 | 20 / 30 | 146 |

- **Données** : pour cette tâche, **25 démos suffisent**, et font aussi bien que 150.
- **Vitesse** : avec 25 démos, le modèle est un peu plus lent en chunks entiers, 173 pas contre 151.
- **Perte** : elle descend plus bas avec 25 démos, jusqu'à 0,0003, ce qui est cohérent avec un jeu plus petit, mieux mémorisé.
- **Test à 10 démos** : en cours au moment de la rédaction, mené par une autre session.

### Limites

- **Tâche simple** : une seule tâche, une seule main, une zone de 10 × 14 cm, un expert scripté parfaitement régulier. Des démos humaines en téléop seront plus variées et plus bruitées. Il en faudra probablement plus.
- **Échantillons** : un seul entraînement par configuration et 30 essais par évaluation. Des écarts de 2 ou 3 réussites ne sont pas significatifs.
- **Réel** : aucun de ces résultats ne vaut encore pour le robot, à cause de l'écart visuel entre sim et réel.

### Recommandations pour le vrai robot

- **Point de départ** : chunks entiers, `--exec-steps 30` sans préfixe, pour la fiabilité. Passer à 10 pas + préfixe de 20 si la réactivité ou la vitesse comptent plus.
- **Nombre de démos** : viser **25 à 50 démos** pour une première tâche plutôt que 200, puis ajuster selon le taux de réussite. Le briefing prévoyait 50 à 200.

