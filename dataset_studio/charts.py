"""Graphique QPainter maison : plusieurs séries sur l'axe des pas, curseur du pas courant, zone de
rognage, bandes des segments étiquetés, clic / glisser = se déplacer dans l'épisode."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .theme import ACCENT, MUTED, SERIES


class SignalChart(QWidget):
    seek = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series: dict[str, np.ndarray] = {}
        self.n = 0
        self.cursor = 0
        self.mark_in = None
        self.mark_out = None
        self.segments: list[tuple] = []          # (début, fin, couleur hex, libellé)
        self.setMinimumHeight(130)
        self.setMouseTracking(False)

    def set_series(self, series: dict, n: int):
        self.series = {k: np.asarray(v, float) for k, v in series.items()}
        self.n = n
        self.update()

    def set_cursor(self, k: int):
        self.cursor = k
        self.update()

    def set_marks(self, a, b):
        self.mark_in, self.mark_out = a, b
        self.update()

    def set_segments(self, segments: list[tuple]):
        self.segments = list(segments)
        self.update()

    # ------------------------------------------------------------------ dessin
    def _plot_rect(self) -> QRectF:
        return QRectF(46, 22, max(10, self.width() - 56), max(10, self.height() - 40))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0a0d12"))
        r = self._plot_rect()
        p.setPen(QPen(QColor("#1c2430"), 1))
        p.drawRect(r)
        if not self.series or self.n < 2:
            p.setPen(QColor(MUTED))
            p.drawText(r, Qt.AlignCenter, "aucune donnée")
            return
        vals = np.concatenate([v[np.isfinite(v)] for v in self.series.values()] or [np.zeros(1)])
        lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        X = lambda k: r.left() + r.width() * k / max(1, self.n - 1)
        Y = lambda v: r.bottom() - r.height() * (v - lo) / (hi - lo)
        # zone de rognage
        if self.mark_in is not None or self.mark_out is not None:
            a = self.mark_in if self.mark_in is not None else 0
            b = self.mark_out if self.mark_out is not None else self.n - 1
            p.fillRect(QRectF(X(a), r.top(), X(b) - X(a), r.height()), QColor(53, 224, 200, 28))
        # segments : fond léger + bande colorée et libellé en bas du graphique
        for a, b, col, label in self.segments:
            c = QColor(col)
            band = QRectF(X(a), r.top(), max(1.0, X(b) - X(a)), r.height())
            c.setAlpha(22)
            p.fillRect(band, c)
            c.setAlpha(230)
            strip = QRectF(X(a), r.bottom() - 12, max(1.0, X(b) - X(a)), 12)
            p.fillRect(strip, c)
            p.setPen(QColor("#0a0d12"))
            p.drawText(strip.adjusted(3, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
        # graduations
        p.setPen(QColor(MUTED))
        for v in (lo + pad, (lo + hi) / 2, hi - pad):
            p.drawText(QRectF(0, Y(v) - 8, 42, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:.2f}")
            p.setPen(QPen(QColor("#151b24"), 1))
            p.drawLine(QPointF(r.left(), Y(v)), QPointF(r.right(), Y(v)))
            p.setPen(QColor(MUTED))
        # séries (sous-échantillonnées à la largeur)
        step = max(1, self.n // int(max(50, r.width())))
        for i, (name, v) in enumerate(self.series.items()):
            col = QColor(SERIES[i % len(SERIES)])
            path, started = QPainterPath(), False
            for k in range(0, len(v), step):
                if not np.isfinite(v[k]):
                    started = False
                    continue
                pt = QPointF(X(k), Y(v[k]))
                if started:
                    path.lineTo(pt)
                else:
                    path.moveTo(pt)
                    started = True
            p.setPen(QPen(col, 1.4))
            p.drawPath(path)
        # légende
        x0 = r.left() + 6
        for i, name in enumerate(self.series):
            col = QColor(SERIES[i % len(SERIES)])
            p.setPen(col)
            txt = name
            v = self.series[name]
            if 0 <= self.cursor < len(v) and np.isfinite(v[self.cursor]):
                txt += f" {v[self.cursor]:.3f}"
            w = p.fontMetrics().horizontalAdvance(txt) + 14
            if x0 + w > r.right():
                break
            p.drawText(QPointF(x0, r.top() - 6), txt)
            x0 += w
        # curseur
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawLine(QPointF(X(self.cursor), r.top()), QPointF(X(self.cursor), r.bottom()))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(r.left(), r.bottom() + 2, r.width(), 16), Qt.AlignRight, f"pas {self.cursor}/{self.n - 1}")

    # ------------------------------------------------------------------ souris
    def _emit(self, x):
        r = self._plot_rect()
        if self.n > 1:
            k = int(round((x - r.left()) / r.width() * (self.n - 1)))
            self.seek.emit(max(0, min(self.n - 1, k)))

    def mousePressEvent(self, e):
        self._emit(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._emit(e.position().x())
