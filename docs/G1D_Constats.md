# Constats — Portage de UnifoLM-WLA-1.0 sur Unitree G1-D

Ce fichier regroupe ce qu'on a vérifié dans le code pour notre cas d'usage. Il complète le briefing [Ma_Reflection.md](Ma_Reflection.md) et le corrige là où le code le contredit.

Mêmes niveaux de confiance que le briefing :

- **[VÉRIFIÉ]** — lu directement dans le code du dépôt.
- **[INFÉRÉ]** — déduction cohérente, à valider.
- **[INCONNU]** — question ouverte.

Sauf mention contraire, tous les constats viennent d'une **lecture statique du code**. Les sections 12 à 15 rapportent des essais **exécutés en sim**. Rien n'a encore été exécuté sur le robot.

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

**Contournement [VÉRIFIÉ]** : geler le VLM par sous-modules au lieu de le geler en bloc. Appliqué dans `g1d_finetune_frozen_vlm.yaml` : 6,87 M paramètres entraînables, projecteur compris.

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

**Corrigé pour nos fine-tunings G1-D** (voir le contournement ci-dessus).

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

**[INFÉRÉ]** La tête est bien une caméra stéréo, ce qui correspond à la binoculaire du G1-D. Nos données et la sim envoient l'œil gauche **brut** sous `head_stereo_left` (voir la section 9) ; la vue rectifiée n'est utile que si on calibre la stéréo.

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
22/22 vérifications passées
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

### Suite

**[FAIT]** Le client sim `sim/wla_client.py` joue le rôle du robot pour le serveur WLA : il rend les vues, exprime les poses dans `B` et applique le chunk avec `track_ee_wla` et `set_gripper` à 30 Hz. Résultats aux sections 12 à 14.

## 8. État des questions ouvertes du briefing

| # | Question | État |
|---|---|---|
| 1 | Effecteur | **Résolu** : Dex1 interne. |
| 2 | Format de nos données de téléop | **Résolu** : `g1d_wla/convert_teleop.py` convertit le JSON xr_teleoperate en LeRobot v3 au format WLA. |
| 3 | 120D = état ⊕ masque ? | **Résolu** : oui, voir la section 3. |
| 4 | Caméra du rig de collecte | **Résolu** : tête stéréo et deux poignets. Nous utilisons l'œil gauche brut. Intrinsèques estimées. |
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
- **Pince** : unité Dex1 brute. La valeur **monte à l'ouverture** : environ 4,5 ouverte au repos, environ 2,4 fermée sur un cube de 4 cm, plage totale de 0 (fermée) à 5,4 (ouverte), valeurs utilisées par le code. La correspondance avec la position des doigts de la sim reste à calibrer.
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
- **[VÉRIFIÉ]** Il connaît les Dex1 câblées en interne : `--ee dex1`, qui pilote les pinces par les moteurs 31 et 33 de la commande bas niveau du G1.
- **[VÉRIFIÉ]** `dex1` exige `--arm G1_29` et interdit `--motion`.
- **[VÉRIFIÉ]** Aucun `--arm` ne correspond au G1-D. Le code ne mentionne ni roues, ni colonne, ni G1-D.
- **[INCONNU]** Le G1-D accepte-t-il la commande bas niveau du G1 29 DoF, avec 35 moteurs sur `rt/lowcmd` ? **À vérifier avant tout essai**, bras dégagés et gains faibles. Une mauvaise correspondance des indices moteurs commanderait les mauvais joints.

### Ce qu'il enregistre

Pour chaque pas, à 30 Hz par défaut :

- **Images** : en JPEG. La tête binoculaire est coupée en deux moitiés **brutes**, gauche et droite. Les poignets sont ajoutés s'ils sont activés dans teleimager.
- **État** : angles **mesurés** des deux bras et position des pinces.
- **Action** : angles **commandés** des bras, sortie de l'IK, et commande des pinces.

Il n'enregistre **pas** les poses effecteur, la pose du buste ni la pose caméra. Notre fork ajoute `body.qpos` (les 35 moteurs, jambes et buste compris) et la commande de base issue des joysticks.

### Ce que produit unitree_lerobot

- **[VÉRIFIÉ]** Le convertisseur public JSON → LeRobot produit l'**ancien format** : un vecteur `observation.state` unique et des vues `cam_left_high`. C'est le format des 11 datasets Dex1 inutilisables par la config WLA.
- **[INFÉRÉ]** Le format WLA vient d'un pipeline Unitree non publié, qui recalcule les poses par cinématique directe. Le champ `recomputed_ee_valid` va dans ce sens, tout comme l'écart constant d'environ 1 cm entre commande et mesure, qui correspond à FK(angles commandés) comparé à FK(angles mesurés).

### Chaîne proposée

1. **Images** : teleimager, avec la tête binoculaire et les deux poignets, à 640×480 par vue.
2. **Téléopération** : si le G1-D est compatible, lancer :
   ```
   python teleop_hand_and_arm.py --arm=G1_29 --ee=dex1 --record --frequency 30 --task-name <nom> --task-goal "<instruction>"
   ```
3. **Conversion** : un convertisseur maison, JSON → LeRobot v3 au format WLA, **écrit** dans `g1d_wla/convert_teleop.py` :
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

Le préfixe rend la replanification fréquente utilisable : à 10 pas par chunk, on passe de 8 à 23 réussites. Imposé à la lettre, il n'égale pas encore la fiabilité des chunks entiers.

**[VÉRIFIÉ]** **Raccord doux**, comme dans la méthode publiée (Black et al., 2025) : au-delà du préfixe, le poids d'inpainting décroît exponentiellement sur quelques pas, en prolongeant la dernière action connue. Option `rtc_soft` du serveur, `--rtc-soft` du client. Mesures sur le modèle à 10 démos :

| Réglage | Réussites | Pas médian |
|---|---|---|
| chunk entier, sans préfixe | 25 / 30 | 157 |
| 10 pas + préfixe de 20 | 18 / 30 | 150 |
| 10 pas + préfixe de 20 + raccord doux de 5 | **25 / 30** | **151** |
| 15 pas + préfixe de 10 + raccord doux de 5 | 13 / 30 | 213 |
| 25 pas + préfixe de 5 | 14 / 30 | 176 |

- **Raccord doux** : avec un préfixe de 20, il rattrape la fiabilité des chunks entiers, en replanifiant trois fois plus souvent et un peu plus vite.
- **[INFÉRÉ] Préfixe court** : un préfixe de 5 ou 10 pas dégrade nettement. Le modèle doit alors reprédire une grande partie du geste en cours sans connaître sa vitesse. Le préfixe doit couvrir l'essentiel du chunk.
- **Vitesse plafond** : environ 150 pas contre 127 pour l'expert, dans tous les bons réglages. Le reste de l'écart vient probablement des reprises à chaque chunk, et de la lenteur prudente d'un modèle qui moyenne ses démos.

### Nombre de démos

**[VÉRIFIÉ]** Même recette, 3 000 pas d'entraînement, sur les 25 premières démos seulement :

| Démos | Réglage | Réussites | Pas médian |
|---|---|---|---|
| 150 | chunk entier | 28 / 30 | 151 |
| **25** | chunk entier | **30 / 30** | 173 |
| 150 | 10 pas + préfixe de 20 | 23 / 30 | 148 |
| 25 | 10 pas + préfixe de 20 | 20 / 30 | 146 |
| **10** | chunk entier | **25 / 30** | 157 |
| 10 | 10 pas + préfixe de 20 + raccord doux de 5 | 25 / 30 | 151 |

- **Données** : pour cette tâche, **25 démos suffisent**, et font aussi bien que 150.
- **Vitesse** : avec 25 démos, le modèle est un peu plus lent en chunks entiers, 173 pas contre 151.
- **Perte** : elle descend plus bas avec 25 démos, jusqu'à 0,0003, ce qui est cohérent avec un jeu plus petit, mieux mémorisé.
- **10 démos**, même recette : **25 / 30** en chunks entiers, en 157 pas médians. Même résultat avec le préfixe et le raccord doux, en 151 pas.

### Non-régression après la brique RECAP (30 septembre 2026)

**[VÉRIFIÉ]** Même checkpoint à 10 démos, mêmes 30 positions, code après la brique RECAP, sans option d'avantage : chunk entier **26 / 30** en 150 pas, contre 25 / 30 en 157 avant ; préfixe de 20 et raccord doux de 5 **24 / 30** en 155 pas, contre 25 / 30 en 151 avant. Les écarts restent dans le bruit, environ ±3 sur 30 : RECAP n'a rien cassé.

### Limites

- **Tâche simple** : une seule tâche, une seule main, une zone de 10 × 14 cm, un expert scripté parfaitement régulier. Des démos humaines en téléop seront plus variées et plus bruitées. Il en faudra probablement plus.
- **Échantillons** : un seul entraînement par configuration et 30 essais par évaluation. Des écarts de 2 ou 3 réussites ne sont pas significatifs.
- **Réel** : aucun de ces résultats ne vaut encore pour le robot, à cause de l'écart visuel entre sim et réel.

### Recommandations pour le vrai robot

- **Point de départ** : chunks entiers, `--exec-steps 30` sans préfixe, pour la fiabilité. Pour plus de réactivité, passer à `--exec-steps 10 --rtc-prefix 20 --rtc-soft 5`, aussi fiable en sim. Ne pas utiliser de préfixe court.
- **Nombre de démos** : en sim, 10 démos donnent déjà 25/30 et 25 démos 30/30. Sur le robot, avec des démos humaines plus variées, viser **25 à 50 démos** pour une première tâche plutôt que 200, puis ajuster selon le taux de réussite. Le briefing prévoyait 50 à 200.

---

## 14. Tâches Novares en sim : saisie et empilement (29 septembre 2026)

### Saisie par la prise peinte

**[VÉRIFIÉ]** Les zones de prise peintes dans mpc_any s'appliquent telles quelles à la sim. Le fichier `configs/projects/usine/novares.zones.json` y est associé au même STL que celui de la sim, avec une md5 identique. Le passage du STL au repère MuJoCo est `v_géom = R(mesh_quat)ᵀ (v_stl − mesh_pos)`.

- **Prise** : une prise peinte, `prise_1`, de 52 mm de large. Les mors serrent les deux extrémités de la paroi courbe. 11 approches sur 24 ne traversent pas la pièce, et l'approche verticale est atteignable par le bras droit.
- **Expert** : il réussit 46 prises sur 50, avec la pièce à ±3 cm et ±20°. Il faut monter la main, faire un transfert articulaire puis descendre. Une rotation cartésienne près de la table faisait balayer la pièce par l'avant-bras.

### Fine-tuning sur la tâche Novares

**[VÉRIFIÉ]** Recette G1-D, 3 000 pas, évaluation sur 30 placements jamais vus. L'expert réussit en 151 pas médians.

| Démos | Réglage | Réussites | Pas médian |
|---|---|---|---|
| 50 | chunk entier | 27 / 30 | 149 |
| 50 | 10 pas + préfixe de 20 | 25 / 30 | 154 |

La pièce Novares, avec sa prise verticale peinte et sa variance de ±3 cm et ±20°, s'apprend aussi bien que le cube. Le modèle atteint la **vitesse de l'expert**. Pour le cube, il restait environ 20 % plus lent, à 151 pas contre 127. Résultats à 25 démos : 22 / 30 en 146 pas ; à 10 démos : 27 / 30 en 161 pas (voir le tableau de MYREADME.md).

### Empilement de deux pièces

**[VÉRIFIÉ]** Les pièces s'emboîtent en sim. La position a été trouvée par recherche géométrique, en prenant la hauteur minimale sans interpénétration :

| Mesure | Valeur |
|---|---|
| Décalage de la pièce du dessus | environ 17 mm vers la gauche et 16 mm plus haut |
| Décalage nul, pour comparaison | 59 mm de haut, posée sur la crête |
| Décalage dans le repère du support | (3, −20, 11) mm, dans `_Novares_Piece1_centered.stack.json`, non versionné |

- **Robustesse** : lâchée de 5 à 30 mm au-dessus, avec ±4 mm et ±5° d'erreur, la pièce retombe presque toujours à moins de 8 mm de la pose emboîtée.
- **Glissement dans la pince** : à la levée, la pièce bouge de 7 à 13 mm et de 6 à 13°. L'expert doit donc replanifier le dépôt avec la pose réellement tenue.

**[VÉRIFIÉ] Blocage** : avec la prise peinte, **toutes** les approches atteignables mettent un doigt, `Link1_1` ou `Link2_1`, contre la pièce du dessous en pose emboîtée. C'est logique : les doigts tiennent les extrémités de la paroi, et ce sont précisément ces zones qui se logent contre la pièce du dessous.

| Variante | Réussites |
|---|---|
| Lâcher à 2 cm, correction du lacet seulement | 3 / 20 |
| Autres combinaisons de hauteur de prise et de lâcher | 0 à 1 / 12 |

Le support est poussé de 7 à 13 mm en médiane.

**[VÉRIFIÉ] Cause du blocage, précisée par l'opérateur (30 septembre)** : les pièces étaient posées **sur le dos**, paroi courbe en bas et plaque en l'air (pose reprise de la scène mpc_any). En réel, elles sont à plat, plaque sur la table, et se prennent **par le côté**, sur les flancs de la paroi.

Scène d'empilement corrigée (`scene_g1d_stack.xml`, pièces retournées de 180°) :
- **Emboîtement** : le décalage du fichier `.stack.json` s'inverse (le support devient le « dessus » de la relation calculée sur le dos). Pièce posée à 1 à 2 cm au-dessus, elle retombe emboîtée à 1,5 mm près, support immobile.
- **Prise** : l'approche est maintenant horizontale, par le côté, comme en réel. L'expert choisit la première prise dont le dépôt est aussi atteignable par le bras droit.
- **Dépôt** : pièce bien orientée au moment d'ouvrir (à 1° près). Lâchée à 2 cm, elle bascule d'environ 25° ; lâchée à 5 mm, elle s'emboîte.

| Variante | Réussites de l'expert |
|---|---|
| Sur le dos (ancienne scène) | 3 / 20 |
| À plat, lâcher à 5 mm | **18 / 60** (10/30 et 8/30 sur deux graines) |

Échecs restants : dépôt hors de portée du bras droit (environ un quart), pièce qui glisse ou tombe de la pince pendant le transport, bascule de 20 à 25° au lâcher. Rapprocher le support n'a pas aidé (5/30 et 6/30).

**[INFÉRÉ]** Un tiers de réussite suffit pour enregistrer des démos (on garde les réussites, comme pour Novares), mais une prise plus stable dans la pince aiderait l'apprentissage.

**[VÉRIFIÉ, sim] Premier modèle d'empilement, 25 démos (30 septembre)** :
- **Démos** : 25 réussites gardées sur 422 essais de l'expert (seules les réussites sans aucun refus d'IK sont gardées, d'où 6 %).
- **Fine-tuning** : 3000 pas depuis le modèle Base, `g1d_sim_stack_n25`.
- **Résultat : 0 / 30.** Le modèle enchaîne bien le geste complet (approche par le côté, transport, dépose), mais sur 12 épisodes diagnostiqués : 9 fois la pièce n'est pas saisie et reste sur la table ; 3 fois elle est posée sur le support mais à 47–92 mm et 37–114° de la pose emboîtée.
- **[VÉRIFIÉ] Ce n'est pas la tolérance de la prise** : pose de prise de l'expert décalée au hasard de r mm, saisie tenue (levée > 5 cm) sur 15 essais :

  | Décalage | Par le dessus (sur le dos) | Par le côté (à plat) |
  |---|---|---|
  | 0 mm | 14 / 15 | 15 / 15 |
  | 2 mm | 14 / 15 | 14 / 15 |
  | 5 mm | 14 / 15 | 14 / 15 |
  | 10 mm | 13 / 15 | 12 / 15 |

- **[INFÉRÉ]** L'échec vient donc de l'apprentissage : 25 démos longues (300 pas, plusieurs phases) triées sur 6 % des essais, qui couvrent mal la variance de placement. Essai : 100 démos, réussites avec refus d'IK acceptées (`--allow-ik-refused`).
- **[VÉRIFIÉ, sim] 100 démos : toujours 0 / 30** (100 réussites gardées sur 585 essais, 3000 pas). Le modèle approche la bonne pièce mais ferme la pince à côté ou la pousse (5 fois sur 30, elle tombe de la table), puis va au-dessus du support à vide. Real-time chunking (10 pas + préfixe 20, raccord 5) : 0 / 12 aussi. Perte d'entraînement comparable aux autres runs (fin à 0,0004), donc pas de sous-apprentissage visible dans la perte.
- **[VÉRIFIÉ] Boucle ouverte** (`sim/open_loop_check.py` : observation enregistrée envoyée au serveur, chunk prédit comparé aux 30 actions enregistrées ; 3 épisodes, une requête tous les 30 ou 15 pas) :

  | Modèle | Épisodes | Écart au 30e pas, médian | « Tenir la pose » |
  |---|---|---|---|
  | Saisie Novares, 10 démos (27/30 en boucle fermée) | d'entraînement | **6,9 mm** | 70 mm |
  | Empilement, 100 démos (0/30) | d'entraînement | **16,0 mm** | 69 mm |
  | Empilement, 100 démos | non vus (jeu n25) | 18,5 mm | 76 mm |

  Le modèle d'empilement reproduit mal ses propres démos : 2 fois moins précis que le modèle Novares, avec des pointes à 5–11 cm pendant le transfert articulaire et le transport, et un instant de fermeture ou d'ouverture de la pince souvent décalé. Presque aussi mauvais sur des démos non vues : ce n'est pas du sur-apprentissage.
- **[INFÉRÉ] Cause probable : sous-apprentissage.** 3000 pas × lot de 2 = 6000 exemples vus, soit environ 4 passages sur les 1 500 images du jeu Novares n10, mais 0,2 passage sur les 30 800 images du jeu d'empilement n100 (0,8 pour n25). La perte d'entraînement, bruitée, ne le montrait pas.
- **[VÉRIFIÉ, sim] 25 démos, 12 000 pas (~3 passages) : toujours 0 / 30**, mais la boucle ouverte passe à **8,3 mm** (contre 16 mm), proche du modèle Novares (6,9 mm). En boucle fermée : 11 / 30 pièces transportées et posées sur le support (60 à 105 mm et 36 à 126° de la pose emboîtée, une à 16 mm), 17 / 30 saisies ratées, 2 pièces tombées.
- **[INFÉRÉ]** Le modèle reproduit maintenant ses démos à peu près aussi bien que celui de Novares : ce qui reste est l'accumulation d'erreurs en boucle fermée (il sort des états vus dans les démos et ne sait pas se rattraper). Pistes : corrections dans la boucle (rollouts + opérateur simulé, type DAgger / RECAP, déjà outillé par `sim/recap_rollouts.py`), trajectoires d'expert plus simples (le transfert articulaire est la phase la plus mal reproduite), démos plus variées (bruit injecté dans l'expert).
- **[VÉRIFIÉ, sim] Itération DAgger 1 (1er octobre)** (`sim/experiments/queue_dagger.sh`) :
  - **Rollouts** : 100 épisodes de la politique n25_12k ; 0 réussite seule, 98 reprises par l'expert (pièce poussée, saisie ratée ou pas de réussite au pas 330), dont **11 seulement réussies** : l'expert rattrape mal les états déviés.
  - **Fine-tuning** : depuis n25_12k, 4000 pas sur 25 démos + 11 corrections.
  - **Résultat : 0 / 30**, boucle ouverte 7,3 mm. En boucle fermée, plutôt moins bien : 6 / 30 pièces posées sur le support (une à 32 mm), 19 saisies ratées, 5 pièces tombées.
- **[INFÉRÉ] Le goulot est l'expert** : il ne réussit que 30 % depuis le départ et 11 % depuis un état dévié. Il produit donc peu de corrections, et des démos triées sur des placements faciles. Avant d'autres itérations, il faut le fiabiliser (portée du bras droit, pièce qui glisse dans la pince, reprise depuis un état quelconque).

**[INFÉRÉ]** L'emboîtement en sim repose aussi sur la décomposition convexe de la pièce en 90 morceaux. Il peut différer de l'emboîtement réel.

---

## 15. Brique RECAP / Delta-0 : amélioration par essais et corrections (29 septembre 2026)

Principe publié par Physical Intelligence, π*0.6 avec RECAP, fin 2025, et repris par Delta-0 :
1. une politique par imitation joue seule ;
2. un opérateur corrige quand elle se trompe ;
3. un modèle de valeur note chaque morceau d'action selon qu'il rapproche du but ;
4. on réentraîne la politique conditionnée par cette note, « bonne » ou « mauvaise » action ;
5. à l'exécution, on lui demande de « bonnes » actions.

### Ce qui est implémenté

**[VÉRIFIÉ sans GPU]** Tout est testé : prompt, dataloader, conversion, étiquetage, faux serveur, smoke test à 22/22.

| Brique | Fichier | Rôle |
|---|---|---|
| Conditionnement par l'avantage | `QWen3.build_qwenvl_inputs`, `QwenMMDiT` | Ligne `Advantage: positive|negative` dans le prompt, **seulement** si l'exemple en porte une. Sinon, le prompt est identique à l'original |
| Données | `config.py`, `single_source_dataset.py`, `dataloader.py` | `advantage_key` et `advantage_dropout` dans la config de données. Une frame vaut 1 si bonne, 0 si mauvaise, −1 si la condition est omise |
| Serveur | `action_server_wbc_msgpack_unitree.py` | `--advantage positive`, ou `obs["advantage"]` par requête |
| Rollouts avec opérateur | `sim/recap_rollouts.py` | La politique joue. L'opérateur simulé, l'expert rejoué depuis l'état courant, prend la main si l'objet est poussé de plus de 3 cm ou si la tâche n'est pas réussie au pas 220. Chaque pas porte `intervention` à 0 ou 1 |
| Modèle de valeur et étiquetage | `g1d_wla/recap.py` | ResNet18 gelé sur la tête et le poignet droit, plus l'état, puis un MLP qui prédit les pas restants avant la réussite ; un échec vaut le maximum. Un morceau de 30 pas est positif s'il gagne au moins 0,5 × 30 pas restants. Les corrections sont toujours positives |
| Conversion | `g1d_wla/convert_teleop.py` | Colonnes `advantage` et `intervention`. Les pas non étiquetés, comme les démos, valent 1 |
| Écriture d'épisode | `sim/sim_episode_writer.py` | Partagé par les démos et les rollouts. `info.success_step` et `info.outcome` sont ajoutés |
| Tâche de test | `sim/sim_tasks.py`, tâche `novares_left` | Pièce 4 à 7 cm à gauche de la zone d'entraînement : c'est l'équivalent de la « cuisine inversée » de Delta-0. La zone « plus loin » (`novares_shift`) a été écartée : le modèle à 10 démos y réussit déjà 30 / 30, contre 7 / 15 à gauche |

### Déroulé de l'expérience

`sim/experiments/queue_recap.sh` enchaîne ces étapes (lancée le 29 septembre à 23 h 07) :
1. **Référence** : la politique à 10 démos, évaluée dans la zone décalée et dans la zone d'origine.
2. **Rollouts** : 40 épisodes dans la zone décalée, avec l'opérateur simulé.
3. **Modèle de valeur et étiquetage.**
4. **Fine-tuning conditionné** : depuis le modèle Base, sur les démos et les rollouts, avec 30 % de condition omise.
5. **Évaluation** avec « Advantage: positive ».
6. **Témoin** : les mêmes données sans conditionnement.

La comparaison entre les étapes 5 et 6 dira si le gain vient du conditionnement par l'avantage ou seulement des données en plus.

### Pour le vrai robot

**[FAIT, non validé sur le robot]** Mode politique avec correction en delta dans xr_teleoperate, `teleop/policy_bridge.py` et option `--policy-uri`. Procédure dans `teleoperation/REAMDEG1D.md`.

- **Pilotage** : la politique WLA pilote les bras par l'IK de la téléop.
- **Correction** : grip maintenu, le déplacement de la manette s'ajoute à la cible, et le pas est enregistré avec `intervention` = 1. La gâchette donne la pince à l'opérateur.
- **Fin d'essai** : `X` ou `Y` gauche termine l'essai, réussi ou raté, et écrit `info.success_step` et `info.outcome`.
- **Repères, vérifiés hors robot** : l'IK de la téléop vise `wrist_yaw` + 0,05 m dans le repère de son modèle G1 à taille verrouillée, ce qui correspond à la base WLA à buste droit. Son URDF a le même poignet que le G1-D. La conversion colle exactement à la FK de l'IK, à 0,00 mm près sur 50 poses aléatoires.
- **Sécurité** : vitesse de la cible bornée, 0,10 m/s par défaut. Les bras tiennent leur pose hors essai.
- **Latence** : une requête au modèle, environ 0,4 s, bloque la boucle. Les bras tiennent leur dernière cible pendant ce temps. Préchargement asynchrone possible plus tard.

### Résultats de la 1re itération (30 septembre 2026)

**[VÉRIFIÉ, sim]** 30 épisodes par évaluation, graine 7, chunk entier de 30 pas :

| Modèle | Zone « à gauche » | Zone d'origine |
|---|---|---|
| Référence : 10 démos | 18 / 30, 170 pas | 27 / 30, 157 pas |
| RECAP, « Advantage: positive » | 14 / 30, 156 pas | 23 / 30, 145 pas |
| Témoin : mêmes données, sans avantage | 13 / 30, 165 pas | 24 / 30, 148 pas |

- **Rollouts** : 17 / 40 réussis par la politique seule, 20 corrections de l'opérateur simulé.
- **Modèle de valeur** : perte d'entraînement quasi nulle, mais erreur de 128 pas en moyenne sur les épisodes tenus hors entraînement (72 à 169 selon le pli). Il apprend par cœur 50 épisodes.
- **Étiquetage** : 53 % des pas de la politique positifs.

**Lecture** :
- RECAP et témoin sont à égalité, à l'écart de bruit près (environ ±3 sur 30) : le conditionnement n'apporte rien ici, ce qui s'explique par des étiquettes proches du hasard.
- Les deux font moins bien que la référence : ajouter des rollouts, dont les pas ratés de la politique, dégrade l'imitation.
- Gain de vitesse léger (156 contre 170 pas), dû aux corrections de l'expert dans les données.

**Pistes** pour une 2e itération : plus de rollouts ; une valeur plus simple (issue de l'épisode, ou pas restants sans image) ; affiner depuis le modèle n10 plutôt que depuis Base ; exclure du témoin les pas négatifs pour isoler l'effet du filtrage.

**[INCONNU]** Plusieurs choix restent ouverts, et le blog de Delta ne les donne pas non plus :
- le seuil d'étiquetage ;
- la taille du jeu de rollouts ;
- le nombre d'itérations.


### Corrections de l'audit (29 septembre 2026)

- **Modèle de valeur** : horizon porté de 300 à 500 pas ; validation croisée à 5 plis, chaque épisode étant noté par un modèle qui ne l'a pas vu ; étiquette par pas sur la fenêtre [t, t+30).
- **Rollouts** : l'issue se juge après 20 pas de tenue ; une prise qui lâche compte comme un échec.
- **File** : cache Arrow vidé avant chaque entraînement, seuil disque ramené à 13 Go.
- **Téléop, mode politique** : la pince ne suit plus la gâchette brute ; la correction est rebasée à chaque chunk ; la borne de vitesse part de la pose mesurée ; une panne du serveur tient les bras au lieu de les renvoyer au repos ; les pas tenus ne sont pas enregistrés ; un essai terminé sans `X`/`Y` est noté `unknown` et ignoré par l'étiquetage. Tests hors robot dans `teleop/test_policy_bridge.py`.

---

## 16. Rotation du buste du G1-D (1er octobre 2026)

**[VÉRIFIÉ, modèle sim]** Dans l'URDF du G1-D, `Yaw_Joint` est en fait le **tangage** du buste (axe y). La rotation gauche-droite est `torso_Joint` (axe z, ±155°), montée **au-dessus** du tangage. Chez le G1, l'ordre est inverse : lacet, roulis, puis tangage de la taille.

**[VÉRIFIÉ]** WLA connaît la rotation : `action.waist_action_joint` = [lacet, roulis, tangage] de la taille du G1. Dans les statistiques d'entraînement, le lacet va de −0,44 à +0,57 rad (q01–q99), mais reste proche de 0 dans 80 % des données (q10–q90 ±0,04).

**Contrat retenu** :
- la base WLA reste **fixe** sous le buste ; le torse dans la base vaut Ry(tangage)·Rz(lacet) (`base_T_torso`) ;
- la taille WLA envoyée et enregistrée est la taille G1 **de même orientation de torse**, calculée exactement (`waist_from_torso`) ; dans l'autre sens, `torso_from_waist` ;
- en sim, la FK/IK des bras suit la rotation mesurée à chaque pas : une cible en base WLA reste tenue à 0,2 mm près pendant que le buste tourne ; le convertisseur reproduit les poses de la sim à 0,6 mm près, buste tourné.

**Téléop** : rotation au joystick droit, rotation prédite appliquée en mode politique (voir `teleoperation/REAMDEG1D.md`). Tests hors robot OK (`test_policy_bridge.py`, test 6). **[INCONNU]** Indice du moteur sur le G1-D, à vérifier (hypothèse 12, comme le lacet de taille du G1_29).

**[VÉRIFIÉ, sim] Pour l'empilement, la rotation n'aide pas l'expert** :

| Rotation du buste au dépôt | Réussites (2 × 30 essais) |
|---|---|
| 0 | 16 / 60 |
| 0,2 rad | 11 / 60 |
| 0,35 rad | 9 / 60 |
| 0,5 rad | 6 / 60 |

Ses échecs viennent surtout de la pièce perdue ou tombée (9 / 30) et du mauvais emboîtement (6 / 30) ; les dépôts hors de portée (5 / 30) ne diminuent pas avec la rotation. Elle reste à 0 par défaut pour l'expert.

---

## 17. Audit du 1er octobre 2026

Quatre relectures indépendantes (téléop, données et modèle, sim, scripts et docs). Constats vérifiés dans le code, puis corrigés (commits « corrections de l'audit du 1/10 »).

**[VÉRIFIÉ] Corrigé, téléop (vrai robot)** :
- image de tête absente pendant un enregistrement : plantage (`bgr` None), puis retour au repos des bras à 20–30 rad/s. Désormais le pas est sauté, et l'arrêt ramène les bras à 0,5 rad/s ;
- connexion au serveur WLA déplacée **avant** le mode debug ;
- cible du mode politique bornée à 5 cm / 0,3 rad de la pose mesurée : un bras bloqué ne voit plus sa consigne avancer sans fin ;
- délai de réponse dépassé : la réponse tardive décalait tous les échanges suivants d'un message ; la connexion est maintenant fermée et rouverte ;
- rotation du buste : indice validé, départ refusé si l'angle mesuré dépasse la borne, consigne tenue à 0,15 rad de la mesure, buste figé en tenue, vitesse bornée en mode politique pour respecter `--policy-max-speed`, correction comptée comme intervention tant qu'elle est non nulle ;
- `--right-only` : la pince gauche se fermait, elle est maintenant figée ; `--sim` avec politique : mêmes filtres d'enregistrement ; `info.body_layout` porte le tangage constant.

**[VÉRIFIÉ] Corrigé, données et RECAP** :
- **étiquetage RECAP faux en fin de tâche** : un morceau devait gagner au moins 15 pas, alors qu'il en restait moins. Les ~15 pas avant la réussite et toute la tenue étaient négatifs, ce qui apprenait au modèle « positif » à éviter de finir. Le gain attendu est maintenant plafonné par les pas restants. Cette erreur a pu peser sur le résultat de la 1re itération RECAP (§15) ;
- `advantage_key` absente du dataset : erreur au lieu d'un entraînement silencieusement sans condition ;
- cache Arrow indexé aussi par le contenu des parquet ;
- convertisseur : tangage constant enregistré prioritaire, valeurs `None` de `body_layout` ignorées ;
- plis du modèle de valeur retrouvés par chemin absolu.

**[VÉRIFIÉ] Corrigé, sim et files** :
- pièce tenue détectée par contacts (le seuil de 3 cm était faux près de la pose emboîtée) ;
- découpe DAgger : les 10 pas de politique gardés avant la reprise étaient l'échec lui-même (pince fermée à vide dans 8 corrections sur 11). `--context 0` par défaut. Cette erreur a pu peser sur l'itération DAgger 1 ;
- boucle ouverte : la taille envoyée était [0, 0, 0] au lieu de la taille enregistrée (le chiffre de 8,3 mm reste valable en comparaison, mais l'observation n'était pas exacte) ;
- verrou GPU atomique, libéré seulement par la file qui l'a pris ; serveur tué à l'arrêt de la file Novares ; rollouts conservés si leur découpage échoue.

**[INCONNU] Non corrigé, à décider** :
- `rtc_soft` (raccord doux du real-time chunking, code de l'autre session) remplit la zone de raccord avec le dernier pas imposé répété, donc « rester immobile » : le robot ralentirait à chaque chunk. À revoir avec l'auteur ;
- `_rtc_prev` du serveur est partagé entre connexions (les clients actuels envoient bien `policy_reset`) ;
- les critères de réussite diffèrent encore légèrement entre l'expert, l'enregistreur, le client et les rollouts ;
- **l'évaluation de l'empilement tire des placements que l'expert lui-même réussit rarement** (4 / 30 sur les placements de l'évaluation) : le 0 / 30 du modèle se lit par rapport à ce plafond.

## 18. Robot réel : novares_box_v2 et novares_stack en boucle ouverte (3 octobre 2026)

`sim/open_loop_check.py` sur les démos réelles, main droite, une requête tous les 30 pas, écart de position au pas 30. Référence « tenir la pose » = ne pas bouger.

| Modèle | Épisodes | Écart médian au pas 30 | Tenir la pose |
|---|---|---|---|
| novares_box (55 démos du 1er oct.) | 0–50 (vus) | 15,2 mm | 96,6 mm |
| novares_box_v2 (121 démos, 8000 pas depuis box) | 0–50 (vus) | 14,0 mm | 96,6 mm |
| novares_box | 70–115 (nouvelles démos, **jamais vues**) | 44,6 mm | 112,3 mm |
| novares_box_v2 | 70–115 (vues à l'entraînement) | 27,2 mm | 112,3 mm |
| novares_stack (111 démos, 20 000 pas depuis box) | 0–100 (vus) | 17,4 mm | 25,3 mm |

- **[VÉRIFIÉ]** v2 garde la précision sur les anciennes démos et s'ajuste aux nouvelles (prises plus à gauche). Il n'y a pas de split : 27 mm est un score sur des démos vues, pas une généralisation. Les 44,6 mm de l'ancien modèle sur ces démos montrent par contre que les positions nouvelles le sortaient de sa distribution.
- **[VÉRIFIÉ]** Empilement : écart 10 mm quand la main bouge peu, 29–38 mm quand elle bouge de plus de 5 cm en 1 s. 16 requêtes sur 138 ont plus de 50 mm d'écart, souvent au départ d'un mouvement (le modèle prédit « rester » ou une autre direction). Les deux bras travaillent (≈ 1,7–1,8 m de trajet par épisode chacun) ; seule la main droite est mesurée ici.
- **[HYPOTHÈSE]** Les gros écarts de l'empilement viennent de l'instant de départ des mouvements, ambigu en boucle ouverte ; seul l'essai robot le tranchera.

**[VÉRIFIÉ] flash-attn 2.8.3 installé** (roue précompilée cu12 / torch 2.8 / cp312, code sm_120 présent). Avec lui, le VLM passe en `flash_attention_2` et la tête DiT en FLASH / FLASH_VARLEN (chemin d'origine d'Unitree) au lieu de SDPA. Sur box_v2, 52 requêtes en boucle ouverte : deux passages SDPA diffèrent de 7,1 mm par requête (médiane), flash et SDPA de 6,3 mm ; l'écart vient du bruit tiré à chaque génération, pas du noyau. Inférence 290 → 283 ms, entraînement 1,66 → 1,65 s/pas : le temps reste dominé par l'optimiseur sur CPU.
