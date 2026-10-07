# LeRobot et UnifoLM-WLA : audit, décision et plan d'intégration

Date : 7 octobre 2026. Dépôt audité : `~/Documents/project/unitree_lerobot/unitree_lerobot/lerobot` = **LeRobot 0.4.1** (Hugging Face, commit du 10/11/2025, avec quelques modifications locales). Le dépôt parent `unitree_lerobot/` contient en plus le code Unitree (G1, Dex1, IK, serveur d'images).

**Point clé : nos datasets convertis sont déjà au format LeRobot v3.0** (`codebase_version: v3.0`). Les outils de dataset de LeRobot s'appliquent donc tels quels.

---

## 1. Ce qui manque à LeRobot aussi (à faire nous-mêmes)

Pas de validation pendant l'entraînement · pas d'early stopping · pas d'EMA des poids · pas d'accumulation de gradient · pas de LoRA générique (seulement dans GR00T) · pas de real-time chunking dans cette version (ajouté plus tard en amont) · pas de classe robot G1 dans LeRobot lui-même.

→ Pour l'early stopping et la validation, LeRobot ne nous fait pas gagner de temps : notre plan (perte de validation dans notre boucle) reste le bon.

---

## 2. Ce qui vaut le coup, par ordre d'intérêt

### ① Outils de dataset — gain immédiat, faible effort
`src/lerobot/datasets/dataset_tools.py` + CLI `lerobot-edit-dataset` (matures, copient et ré-indexent parquet, vidéos et métadonnées) :

| Outil | Pour nous |
|---|---|
| `split_dataset(ds, splits={"train": [...], "val": [...]})` | **découpe entraînement / validation** directement sur le dataset converti (par fraction ou par liste d'épisodes) — aujourd'hui on la fait en amont, sur les épisodes bruts |
| `delete_episodes` | retirer des épisodes d'un dataset converti sans reconvertir |
| `merge_datasets` / `aggregate_datasets` | fusionner plusieurs datasets en un seul (vérifie la compatibilité des métadonnées) |
| `modify_features` (API Python) | ajouter / retirer une colonne (ex. étiquette de progression, avantage RECAP) sans reconvertir |
| `lerobot-dataset-viz` (Rerun) | visualiser un dataset converti, y compris à distance (`--mode distant`) |

**Proposition** : utiliser `split_dataset` pour nos futures validations (plus simple que dupliquer les épisodes bruts), et `delete_episodes` / `merge_datasets` depuis le Dataset Studio (bouton « Exporter » → dataset converti).

### ② Augmentation d'images — probablement le meilleur gain de performance
`src/lerobot/datasets/transforms.py` (torchvision v2, autonome, facile à copier) : `ImageTransforms`, `RandomSubsetApply` (tire k transformations pondérées par image), `SharpnessJitter`. Défauts : luminosité / contraste (0,8–1,2), saturation (0,5–1,5), teinte (±0,05), netteté (0,5–1,5), petite affine (±5°, translation 5 %). Appliqué à l'entraînement, par caméra, images seulement.

**Notre entraînement n'a aujourd'hui aucune augmentation d'image.** Avec 100 à 200 démos, c'est le levier classique contre le surapprentissage et pour la robustesse (éclairage, flou léger — cf. la caméra floue du 6/10).

⚠ Pour le **visual prompt**, ne pas modifier la teinte (le vert / rouge porte l'information) : garder luminosité / contraste / netteté / petite affine, pas `hue`. Et appliquer l'affine de la même façon aux masques… ici les couleurs sont déjà dessinées dans l'image, donc c'est automatique.

**Proposition** : porter `ImageTransforms` dans notre dataloader (`unifolm_wla/dataloader`), activable par config, sans teinte pour les modèles VP. À tester avec la validation (même entraînement avec / sans augmentation).

### ③ Statistiques et normalisation
`compute_stats.py` (`RunningQuantileStats`, `aggregate_stats`), `augment_dataset_quantile_stats.py`, normalisation `QUANTILES` (q01 / q99 → [-1, 1]) dans `processor/normalize_processor.py`. On fait déjà la même chose (minmax_q avec les statistiques d'Unitree). Utile seulement si un jour on recalcule nos statistiques.

### ④ Têtes flow matching de référence (pour comparer, pas pour remplacer)
- **GR00T N1.5** (`policies/groot/action_head/flow_matching_action_head.py`, `cross_attention_dit.py`) : DiT avec AdaLN (temps) et cross-attention vers les tokens du VLM, encodeurs par robot (`unitree_g1` est un embodiment connu). Le plus proche de notre MM-DiT ; LoRA intégré.
- **π0 / π0.5** : tirage du temps Beta, intégration d'Euler en 10 pas, adaRMS (π0.5), état discrétisé dans le prompt (π0.5).
- **SmolVLA** : VLM gelé + expert plus petit, **cache KV du préfixe** à l'inférence.

**Idée à retenir** : le cache KV / la réutilisation du contexte pendant les 4 pas d'Euler — on calcule déjà le VLM une seule fois, donc gain modeste ; à garder en tête si on optimise la latence.

### ⑤ Actions relatives
`processor/relative_action_processor.py` : conversion action ↔ action − état sur des dimensions masquées. On a déjà nos poses relatives (plus complètes : rotations) — rien à reprendre.

### ⑥ Inférence asynchrone
`async_inference/` (serveur / client gRPC + pickle) : déclenche la requête suivante quand la file d'actions descend sous un seuil (`chunk_size_threshold`, 0,5) et fusionne le recouvrement ancien / nouveau chunk (`weighted_average` 0,3 / 0,7, `latest_only`…). `ACTTemporalEnsembler` : moyenne pondérée exponentielle des chunks.

Notre serveur websocket + msgpack et notre real-time chunking font mieux ; le transport gRPC / pickle n'a pas d'intérêt. **À reprendre éventuellement** : la fusion `weighted_average` comme alternative simple, et le seuil de déclenchement, pour comparer avec notre RTC.

### ⑦ Boucle d'entraînement
Ordonnanceur `cosine_decay_with_warmup` (s'ajuste au nombre de pas), sauvegarde avec état complet (optimiseur, ordonnanceur, graines aléatoires) pour une reprise exacte, `WandBLogger`, `MetricsTracker` (échantillons, épisodes, **passages sur les données**). On a l'essentiel ; le suivi des passages sur les données serait utile dans nos journaux.

### ⑧ Évaluation
- `lerobot-record --policy.path` : enregistrer des essais de politique comme dataset — on le fait déjà avec la téléop.
- HIL-SERL (`rl/`) : RL avec intervention humaine, classifieur de récompense — conçu pour de petits bras en espace effecteur, lourd à adapter à un VLA. Notre RECAP est plus adapté.
- Dépôt parent : `eval_robot/eval_g1_dataset.py` trace action prédite contre vérité terrain par dimension — bonne idée pour **visualiser** nos contrôles hors ligne (aujourd'hui on n'a que des chiffres).

### ⑨ Divers
`EpisodeAwareSampler` (ignorer les N premiers / derniers pas de chaque épisode — utile pour les chunks de 30 pas en fin d'épisode), `StreamingLeRobotDataset` (si les données grossissent), `TimerManager` (mesures de temps), Rerun.

---

## 3. Décision : on garde notre dépôt, on reprend seulement ce qui sert

**On ne porte pas WLA dans LeRobot.** Raisons :

- **Mémoire GPU** : WLA ne tient sur une RTX 5090 (24 Go) que grâce à DeepSpeed ZeRO-2 avec l'optimiseur sur le CPU ; LeRobot n'est configuré que pour DDP (pas de DeepSpeed / FSDP).
- **Réécriture lourde** : modèle (VLM gelé + MM-DiT) en `PreTrainedPolicy`, format unifié 60 / 54 avec masques, normalisation aux statistiques d'Unitree, poses relatives, conditionnement par l'avantage (RECAP), visual prompt — des semaines, avec le risque de casser ce qui marche.
- **Instabilité** : LeRobot change souvent d'API (v2.1 → v3, refonte des processors) ; on subirait chaque mise à jour.
- **Spécifique au robot** : téléop G1-D, serveur websocket / msgpack, real-time chunking, RECAP, Dataset Studio, Inference Studio n'ont pas d'équivalent G1 dans LeRobot.

**Règles d'intégration :**

| Type | Comment | Exemples |
|---|---|---|
| Petit module autonome | **copié** dans notre dépôt, licence Apache 2.0, en-tête citant la source et la version | augmentation d'images (`datasets/transforms.py`) |
| Outil de données | **appelé** depuis l'env `unitree_lerobot` (où LeRobot 0.4.1 est installé), pas de dépendance ajoutée à `.venv` | `split_dataset`, `merge_datasets`, `delete_episodes`, `lerobot-dataset-viz` |
| Idée / méthode | **réimplémentée** chez nous, adaptée | courbes du contrôle hors ligne, fusion `weighted_average` des chunks, suivi des passages sur les données |
| Modèles concurrents | **LeRobot à côté**, comme banc de comparaison sur nos datasets v3 | SmolVLA, π0.5, GR00T entraînés sur les mêmes données |

## 4. Recommandations

| Priorité | Action | Effort | Gain attendu |
|---|---|---|---|
| 1 | **Augmentation d'images** dans notre dataloader (sans teinte pour le VP), testée avec la validation | ½ journée | robustesse, moins de surapprentissage |
| 2 | **`split_dataset`** pour les validations futures, au lieu de dupliquer les épisodes bruts | 1 h | simplicité |
| 3 | **Courbes du contrôle hors ligne** par dimension (idée de `eval_g1_dataset.py`) | 2 h | voir *où* le modèle se trompe |
| 4 | `delete_episodes` / `merge_datasets` accessibles depuis le Dataset Studio | ½ journée | moins de reconversions |
| 5 | Fusion `weighted_average` des chunks comme option, à comparer au RTC | 2 h | fluidité, à mesurer |
| 6 | Banc de comparaison : SmolVLA / π0.5 / GR00T sur un de nos datasets v3, avec LeRobot | 1 jour + entraînements | savoir si WLA est le bon modèle pour nos tâches |
| — | Validation, early stopping, EMA | à faire nous-mêmes | (LeRobot ne les a pas) |

## 5. Limites de cet audit

- Lecture du code et de la documentation de LeRobot (exploration par chemins, classes, options, CLI) ; **aucun outil n'a été exécuté** sur nos données.
- Version auditée : 0.4.1 (novembre 2025). Les versions plus récentes ont ajouté des fonctions (real-time chunking, peut-être d'autres) non examinées ici.
- Parties survolées faute d'intérêt : politiques RL (TDMPC, SAC), robots et téléopérations de petits bras (SO100, Koch, LeKiwi).
