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

À valider au démarrage d'un entraînement avec la sortie de `print_trainable_parameters`. Les paramètres `robot_state_projector.net.*` doivent y apparaître comme entraînables, et le reste du VLM comme gelé.

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
- **Taille** : les 3 articulations de taille du G1 n'existent pas sur le G1-D. Masquer ce slot dans l'état et ignorer la valeur prédite.
- **Jambes** : masquer les slots jambes dans l'état et ignorer leurs prédictions.

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

### Ce qui colle déjà avec WLA

- **Cadence** : contrôle à 30 Hz, comme les chunks WLA de 30 pas par seconde.
- **Interface cartésienne** : `track_tcp` prend une pose 4×4 absolue. C'est exactement ce que le serveur WLA renvoie après composition.
- **Format d'image** : les rendus en 640×480 ont le même rapport 4:3 que l'entrée WLA en 448×336.
- **Trois vues** : une vue haute et deux vues de poignet, comme les trois rôles WLA.

### Écarts à combler pour brancher WLA

| Point | Sim | WLA attend | État |
|---|---|---|---|
| Repère des poses | monde MuJoCo | repère base `B` du G1, au bassin | **[INCONNU]** transformation monde → `B` à définir |
| Point effecteur | `gripper_base_link` + 10,5 cm le long des doigts | pose `ee_pose_gripper_base` du G1 | **[INCONNU]** quel point et quels axes Unitree utilise |
| Caméra haute | `torso_rgbd`, monoculaire sur le torse, fovy 65° | vue gauche rectifiée de la stéréo de tête | **[INCONNU]** caméra réelle de notre G1-D |
| Caméras poignet | fovy 110° | intrinsèques non lues | **[INCONNU]** |
| Pince | fermeture de 0 à 1 | valeur Dex1 brute, de 0 à 5,5 environ, médiane 3,3 | **[INCONNU]** sens d'ouverture |
| Base roulante | aucun actionneur de roue | commande de base en vitesses | non testable en sim |
| Colonne | verrouillée en butée basse | hauteur du bassin G1 autour de 0,74 m | correspondance à définir |

**Pourquoi le point effecteur compte.** Les actions sont relatives à l'ancre. Un décalage fixe du point effecteur, ou une rotation fixe de ses axes, change les translations relatives dès que la pince tourne. Il faut utiliser le même point et les mêmes axes que le G1 des datasets.

### Indice sur le repère base [INFÉRÉ]

Dans les statistiques Dex1, la médiane de la pose effecteur gauche vaut environ x = 0,33 m, y = 0,15 m et z = 0,14 m dans le repère base. La hauteur médiane du bassin G1 vaut 0,74 m. La main de travail du G1 est donc à environ 0,88 m du sol.

Dans la sim, la pose de travail gauche est à x = 0,35 m, y = 0,15 m et z = 0,90 m dans le repère monde. Les deux sont très proches. Un repère `B` placé à environ 0,76 m au-dessus du sol, sous le torse, mettrait nos poses dans la même distribution que le G1. À confirmer avec la position du torse dans le monde MuJoCo et avec l'URDF du G1.

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
| 4 | Caméra du rig de collecte | Partiel : tête stéréo, vue gauche rectifiée, et deux poignets. Intrinsèques inconnues. |
| 5 | Repère base du G1-D | Ouvert : comparer les URDF G1 et G1-D. |
| 6 | Base fixe ou roulante | **Résolu** : roulante, pilotable par la commande de base, voir la section 5. |

Nouvelles questions :

| # | Question | Comment trancher |
|---|---|---|
| 7 | Le gel par sous-modules libère-t-il bien le projecteur seul ? | Lancer un fine-tuning court et lire la liste des paramètres entraînables. |
| 8 | Correspondance entre la hauteur G1 et la position de la colonne G1-D | Comparer les URDF à hauteur de caméra égale. |
| 9 | Écart commande/mesure dans les données officielles | Charger un épisode Dex1 et comparer les deux poses à t. |
| 10 | Point et axes de l'effecteur G1 dans les datasets | URDF du G1 Dex1 et code d'enregistrement Unitree. |
| 11 | Sens et unité de la pince Dex1 dans les datasets | Tracer la pince sur un épisode avec prise. |
| 12 | Caméra haute réelle du G1-D : stéréo de tête ou RGBD sur le torse ? | Demander à l'utilisateur. |
