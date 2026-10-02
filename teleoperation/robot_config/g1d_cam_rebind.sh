#!/bin/sh
# G1-D (téléop innov8) : rattache le pilote uvcvideo aux caméras de teleimager avant son démarrage.
# Quand teleimager s'arrête, il relâche les caméras puis veut recharger uvcvideo, ce qui échoue tant que
# la RealSense d'un autre projet (rs_stream.py) l'utilise : les caméras restaient SANS pilote.
# Recherche par NUMÉRO DE SÉRIE (pas par port USB) ; la RealSense n'est pas touchée.
# Installé par teleimager.service.d/rebind.conf ; à retirer : supprimer ce fichier et le drop-in.
SERIALS="01.00.00 JR0001 JR0002"
for dev in /sys/bus/usb/devices/*; do
    [ -f "$dev/serial" ] || continue
    s=$(cat "$dev/serial")
    case " $SERIALS " in *" $s "*) ;; *) continue ;; esac
    for itf in "$dev"/"$(basename "$dev")":*; do
        [ -f "$itf/bInterfaceClass" ] || continue
        [ "$(cat "$itf/bInterfaceClass")" = "0e" ] || continue          # 0e = vidéo
        name=$(basename "$itf")
        [ -e "/sys/bus/usb/drivers/uvcvideo/$name" ] && continue
        timeout 5 sh -c "echo $name > /sys/bus/usb/drivers/uvcvideo/bind" 2>/dev/null \
            && echo "g1d_cam_rebind : $name ($s) rattaché" || echo "g1d_cam_rebind : $name ($s) échec"
    done
done
exit 0
