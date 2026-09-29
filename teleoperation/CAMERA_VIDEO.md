# 📹 Vidéo / Caméras G1 — doc des changements

Tout ce qu'on a modifié et compris sur le flux caméra du téléop G1 (teleimager + affichage VR + enregistrement).

---

## 1. Architecture du flux vidéo

Deux machines :
- **Robot (PC2, Jetson)** — `192.168.123.164` (filaire) + `wlan0` wifi (DHCP, SSID `Innov8_Robot`). Fait tourner **teleimager** (serveur d'image), branché aux caméras USB.
- **PC (laptop)** — fait tourner `teleop_hand_and_arm.py` (contrôle + serveur VR **Vuer** sur `:8012`).
- **Casque VR** — sur le wifi, parle à Vuer via `vuer.ai`.

Deux modes possibles pour amener l'image caméra dans le casque :

| Mode | Chemin | Le casque doit joindre… | Choisi ? |
|---|---|---|---|
| **WebRTC** | casque ⟶ **directement** robot `:60001` | le réseau du robot | ❌ (le casque wifi ne joignait pas bien le robot → *connection refused*) |
| **ZMQ** | robot `:55555` ⟶ **PC** (`image_client`) ⟶ casque via Vuer `:8012` | seulement le **PC** | ✅ **retenu** |

➡️ On a basculé la caméra tête en **ZMQ** : l'image transite par le PC et arrive au casque par le canal Vuer qui marche déjà. Le casque n'a besoin de parler qu'au PC.

> Le code choisit le chemin via `xr_need_local_img = not (display_mode == 'pass-through' or head_camera.enable_webrtc)`.
> Avec `enable_webrtc: false` → `xr_need_local_img = True` → le PC tire les frames (ZMQ) et les rend au casque.

---

## 2. Réseau & ports

- **Contrôle robot (DDS)** : lien **filaire** `enx0c3796e0bc5b` (192.168.123.x) → `--network-interface=enx0c3796e0bc5b`.
- **Caméras** : on pointe le PC vers l'IP **filaire** du robot (stable) → `--img-server-ip=192.168.123.164`.
  (Le wifi du robot `10.3.8.245` marche aussi, mais le filaire est plus stable pour tirer le flux.)
- teleimager écoute sur `0.0.0.0` → joignable sur **toutes** les interfaces du robot (filaire + wifi).

| Port | Rôle |
|---|---|
| `60000` | Requête de config caméra (ZMQ REP) |
| `55555` | Flux **tête** (ZMQ) |
| `55556` | Flux **poignet gauche** (ZMQ) |
| `55557` | Flux **poignet droit** (ZMQ) |
| `60001` | WebRTC tête (désactivé désormais) |

---

## 3. Config caméra côté ROBOT

Fichier : `/home/unitree/unitree_eai_environment/service/teleimager/cam_config_server.yaml`
(sauvegardes faites : `cam_config_server.yaml.bak.*`)

État final :

| Caméra | serial | résolution | `enable_zmq` | `enable_webrtc` |
|---|---|---|---|---|
| `head_camera` | `01.00.00` | 480×1280 (binoculaire) | ✅ true | ❌ **false** |
| `left_wrist_camera` | `0001` | 480×640 | ✅ true | false |
| `right_wrist_camera` | `0002` | 480×640 | ✅ true | false |

Changements appliqués :
1. **head_camera : `enable_webrtc: false`** (zmq gardé) → bascule en chemin ZMQ (cf. §1).
2. **Caméras poignet** : d'abord désactivées (elles n'étaient pas branchées → faisaient planter tout le serveur : `Cannot find UVCCamera serial 0001/0002`), puis **réactivées en ZMQ** une fois branchées.

> Le serveur retrouve chaque caméra par son **`serial_number`** (prioritaire sur `video_id`), donc robuste au renumérotage des `/dev/video*` après reboot.

teleimager tourne en **service systemd** (`teleimager.service`, auto-start, env `tv`).

---

## 4. Modifs code côté PC

Fichier : `Tele_OP/xr_teleoperate/teleop/teleop_hand_and_arm.py`

### 4.1 Bug corrigé — affichage ZMQ
`get_head_frame()` renvoie un objet **`TeleImage`**, pas un tableau numpy. Le code passait l'objet brut à
`render_to_xr()` → crash `cv2.error: cvtColor ... src is not a numpy array`.
**Correctif** : utiliser `head_img.bgr` (+ garde anti-`None`) aux 2 points de rendu.
(Bug invisible en WebRTC car `render_to_xr` est ignoré dans ce mode — d'où le fait que personne ne l'avait vu.)

### 4.2 Fonctionnalité ajoutée — vignettes poignets dans le casque (PiP)
- Fonction **`compose_xr_image()`** : incruste les flux poignets en **vignettes** (coins **haut** gauche/droit, dans chaque œil pour la stéréo) sur une **copie** de l'image tête.
- Flag **`--wrist-pip`** (activé par défaut) / **`--no-wrist-pip`** pour désactiver.
- ⚠️ L'incrustation se fait sur une copie → **les images enregistrées restent propres** (sans vignettes).
- Marche uniquement en mode d'affichage **ZMQ** (le code d'origine n'envoie que la tête au casque).

---

## 5. Problèmes rencontrés & causes

| Symptôme | Cause | Solution |
|---|---|---|
| Pas de caméra du tout | `teleimager.service` en **crash-loop** : caméras poignet non branchées + port `60000` pris par **slamware** ; à chaque crash `modprobe -r uvcvideo` → plus de `/dev/video*` | Désactiver les caméras absentes ; libérer `60000` (reboot) ; `modprobe uvcvideo` |
| Vidéo noire / `connection refused` dans le casque | Casque (wifi) ne joint pas le WebRTC du robot `:60001` | Bascule **ZMQ** (image via le PC) |
| WebSocket qui tombe à l'entrée en VR (`Websocket session is missing`) | Bug du front-end Vuer servi en local | Passer par **`vuer.ai`** (front-end hébergé) |
| `cvtColor ... not a numpy array` | Bug code (`TeleImage` au lieu de `.bgr`) | Corrigé (§4.1) |
| `re_grpc_server ... Arrow IPC` pendant l'enregistrement | **Rerun** (visualiseur), purement cosmétique | Ignorer, ou lancer avec `--headless` |

---

## 6. Commandes utiles (diagnostic / maintenance)

Sur le **robot** (`ssh unitree@192.168.123.164`, mot de passe non versionné) :
```bash
# état + redémarrage serveur d'image
systemctl is-active teleimager.service
sudo systemctl restart teleimager.service
journalctl -u teleimager.service -n 20 --no-pager      # attendre "... is ready" + "Running..."

# lister les caméras détectées (serial / video / chemin) — stopper le service d'abord
sudo systemctl stop teleimager.service
teleimager-server --cf        # (dans l'env tv)
sudo systemctl start teleimager.service

# driver caméra absent (/dev/video* vides)
sudo modprobe -r uvcvideo && sudo modprobe uvcvideo

# ports en écoute
sudo ss -tlnp | grep -E ':(60000|55555|55556|55557)'
```

Sur le **PC** — tester la réception des flux sans lancer la VR :
```python
from teleimager.image_client import ImageClient
c = ImageClient(host="192.168.123.164", request_bgr=True); c.get_cam_config()
print(c.get_head_frame().bgr.shape)        # (480, 1280, 3)
print(c.get_left_wrist_frame().bgr.shape)  # (480, 640, 3)
print(c.get_right_wrist_frame().bgr.shape) # (480, 640, 3)
```

---

## 7. État vérifié ✅

- 3 caméras détectées et reçues sur le PC : tête `(480,1280,3)`, poignets `(480,640,3)`.
- Affichage casque : caméra tête + 2 vignettes poignets (PiP).
- Enregistrement intègre : `data.json` + JPEG valides (4 images/frame : tête G/D + 2 poignets), images **sans** vignettes.

> Voir [REAMDEG1D.md](REAMDEG1D.md) pour les commandes de démarrage dans l'ordre.
