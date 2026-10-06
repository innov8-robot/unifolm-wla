"""Frise de l'épisode, sous la vidéo : segments étiquetés (couleur + nom), sélection [début, fin],
pas corrigés par l'opérateur (intervention), curseur de lecture.

Souris : clic = aller à ce pas · glisser = sélectionner une plage (= début / fin, comme I / O) ·
clic droit sur un segment = menu (supprimer, reprendre ses bornes).
"""
import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

from .theme import ACCENT, MUTED


class Timeline(QWidget):
    seek = Signal(int)
    selection = Signal(object, object)          # (début, fin) ; None = pas de borne
    segment_action = Signal(int, str)           # (indice du segment, "delete" | "marks")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n = 0
        self.cursor = 0
        self.mark_in = self.mark_out = None
        self.segments: list[tuple] = []         # (début, fin, couleur hex, libellé)
        self.intervention = None                # tableau (T,) de 0 / 1, ou None
        self._drag0 = None
        self.setMinimumHeight(54)
        self.setMaximumHeight(54)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Clic : aller à ce pas · glisser : sélectionner une plage (début / fin) · "
                        "clic droit sur un segment : supprimer / reprendre ses bornes")

    # ------------------------------------------------------------------ données
    def set_episode(self, n: int, segments=(), intervention=None):
        self.n, self.segments, self.intervention = n, list(segments), intervention
        self.update()

    def set_cursor(self, k: int):
        self.cursor = k
        self.update()

    def set_marks(self, a, b):
        self.mark_in, self.mark_out = a, b
        self.update()

    def set_segments(self, segments):
        self.segments = list(segments)
        self.update()

    # ------------------------------------------------------------------ dessin
    def _r(self) -> QRectF:
        return QRectF(8, 6, max(10, self.width() - 16), self.height() - 14)

    def _x(self, k: float) -> float:
        r = self._r()
        return r.left() + r.width() * k / max(1, self.n - 1)

    def _k(self, x: float) -> int:
        r = self._r()
        return int(round(max(0.0, min(1.0, (x - r.left()) / r.width())) * (self.n - 1)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0a0d12"))
        r = self._r()
        p.fillRect(r, QColor("#151b24"))
        if self.n < 2:
            p.setPen(QColor(MUTED))
            p.drawText(r, Qt.AlignCenter, "aucun épisode")
            return
        band = QRectF(r.left(), r.top() + 6, r.width(), r.height() - 16)
        f = QFont(p.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        for a, b, col, label in self.segments:
            c = QColor(col)
            seg = QRectF(self._x(a), band.top(), max(2.0, self._x(b) - self._x(a)), band.height())
            c.setAlpha(200)
            p.fillRect(seg, c)
            p.setPen(QColor("#0a0d12"))
            p.drawText(seg.adjusted(4, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
        if self.intervention is not None and len(self.intervention) == self.n:   # pas corrigés : tirets
            p.setPen(QPen(QColor("#e3b341"), 2))
            idx = np.where(np.asarray(self.intervention) > 0.5)[0]
            for k in idx[:: max(1, len(idx) // 400)]:
                x = self._x(k)
                p.drawLine(int(x), int(r.bottom() - 7), int(x), int(r.bottom() - 2))
        if self.mark_in is not None or self.mark_out is not None:                  # sélection
            a = self.mark_in if self.mark_in is not None else 0
            b = self.mark_out if self.mark_out is not None else self.cursor
            lo, hi = min(a, b), max(a, b)
            sel = QRectF(self._x(lo), r.top(), max(2.0, self._x(hi) - self._x(lo)), r.height())
            p.fillRect(sel, QColor(53, 224, 200, 45))
            p.setPen(QPen(QColor(ACCENT), 1.5))
            p.drawRect(sel)
        p.setPen(QPen(QColor("#ffffff"), 2))                                       # curseur
        x = self._x(self.cursor)
        p.drawLine(int(x), int(r.top() - 3), int(x), int(r.bottom() + 3))

    # ------------------------------------------------------------------ souris
    def _segment_at(self, x) -> int | None:
        k = self._k(x)
        hits = [i for i, (a, b, _c, _l) in enumerate(self.segments) if a <= k <= b]
        return min(hits, key=lambda i: self.segments[i][1] - self.segments[i][0]) if hits else None

    def mousePressEvent(self, e):
        if self.n < 2:
            return
        x = e.position().x()
        if e.button() == Qt.RightButton:
            i = self._segment_at(x)
            if i is None:
                return
            a, b, _c, label = self.segments[i]
            m = QMenu(self)
            act_del = m.addAction(f"Supprimer le segment « {label} »  ({a} → {b})")
            act_marks = m.addAction("Reprendre ses bornes (sélection)")
            chosen = m.exec(e.globalPosition().toPoint())
            if chosen is act_del:
                self.segment_action.emit(i, "delete")
            elif chosen is act_marks:
                self.segment_action.emit(i, "marks")
            return
        self._drag0 = self._k(x)
        self.seek.emit(self._drag0)

    def mouseMoveEvent(self, e):
        if self._drag0 is None or not (e.buttons() & Qt.LeftButton):
            return
        k = self._k(e.position().x())
        if abs(k - self._drag0) >= 2:
            self.selection.emit(min(k, self._drag0), max(k, self._drag0))
        self.seek.emit(k)

    def mouseReleaseEvent(self, e):
        self._drag0 = None
