"""Qt overlays painted on top of a plot canvas.

fastplotlib draws the data; these draw the things that have to sit *on* the
data and stay legible — per-trace identifiers and an amplitude calibration bar.
Qt rather than GPU graphics because text is the point: crisp glyphs at arbitrary
sizes, hinted and antialiased by the platform, with no vertex churn as the view
scrolls.

Each is parented to the figure's Qt widget (the same trick
:class:`~phosphor.channel_plot.ChannelPlotWidget` already uses for its range
label), click-through so the plot keeps receiving wheel and hover events, and
told where the traces are by the owning plot — see
``ChannelPlotWidget._sync_overlays``. They hold no reference to the plot, which
is what keeps them reusable and testable.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = ["ChannelLabelOverlay", "ScaleBarOverlay"]

# Font size bounds in pixels. Between them the label is scaled to about half the
# per-trace spacing, so it reads as centred in its row.
MIN_LABEL_FONT_PX = 8
MAX_LABEL_FONT_PX = 40

LABEL_TEXT_COLOR = (205, 205, 210)
LABEL_CHIP_COLOR = (25, 25, 30, 185)

# Scale bar layout, in canvas pixels.
SCALE_BAR_RIGHT_MARGIN = 14
SCALE_BAR_SERIF_HALF = 4
SCALE_BAR_TEXT_GAP = 6
SCALE_BAR_MIN_DRAW_PX = 1.0
SCALE_BAR_FONT_PX = 10


class _CanvasOverlay(QtWidgets.QWidget):
    """Shared setup: transparent, click-through, no background."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Click-through so the plot keeps receiving wheel/hover, and no opaque
        # background so only the painted marks show.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)


class ChannelLabelOverlay(_CanvasOverlay):
    """Per-trace identifiers, drawn beside each channel's baseline.

    Feed it the *full* label list via :meth:`set_labels` and the current
    scroll/zoom window via :meth:`set_view`; it indexes the list by absolute
    channel so scrolling needs no reslicing by the caller.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: list[str] = []
        self._offset = 0
        self._n_visible = 0
        self._top_down = True
        # Affine canvas-y mapping supplied by the owner:
        #   screen_y(world_y) = y0 + slope * world_y
        # None until the first render can measure the camera.
        self._y0: float | None = None
        self._slope: float | None = None
        self._z_scale = 1.0

    def set_labels(self, labels: list[str]) -> None:
        """Set the full per-absolute-channel label list."""
        labels = list(labels)
        if labels == self._labels:
            return
        self._labels = labels
        self.update()

    def set_view(
        self,
        offset: int,
        n_visible: int,
        top_down: bool,
        y0: float | None = None,
        slope: float | None = None,
        z_scale: float = 1.0,
    ) -> None:
        """Update the visible window and the world→screen-y mapping.

        ``y0``/``slope`` come from the plot's actual camera, so labels land on
        the trace baselines rather than on a re-derived guess. Without them the
        overlay falls back to reconstructing the layout analytically, which is
        right until someone pans or zooms. Repaints only on change.
        """
        state = (offset, n_visible, top_down, y0, slope, z_scale)
        if state == (self._offset, self._n_visible, self._top_down, self._y0, self._slope, self._z_scale):
            return
        self._offset, self._n_visible, self._top_down = offset, n_visible, top_down
        self._y0, self._slope, self._z_scale = y0, slope, z_scale
        self.update()

    def _row_geometry(self, n: int, h: int) -> tuple[float, float]:
        """``(y0, slope)`` mapping world-y to canvas-y for the current view.

        Prefers the measured camera mapping. The fallback reproduces the
        autoscaled layout: rows at ``y = row * z_scale`` with a symmetric margin
        of ``max((n - 1) * z_scale * 0.05, 0.5)``, filling the height.
        """
        z = self._z_scale or 1.0
        if self._y0 is not None and self._slope is not None and self._slope != 0.0:
            return self._y0, self._slope
        top = (n - 1) * z
        margin = max(top * 0.05, 0.5)
        span = top + 2 * margin
        # Higher world-y maps nearer the top, so the slope is negative.
        slope = -h / span
        y0 = (top + margin) / span * h
        return y0, slope

    def paintEvent(self, event) -> None:
        n = self._n_visible
        h = self.height()
        if not self._labels or n <= 0 or h <= 0:
            return

        z = self._z_scale or 1.0
        y0, slope = self._row_geometry(n, h)
        row_span_px = abs(slope * z)
        if row_span_px <= 0:
            return
        font_px = max(MIN_LABEL_FONT_PX, min(round(float(0.5 * row_span_px)), MAX_LABEL_FONT_PX))

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        font = QtGui.QFont()
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setPixelSize(font_px)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # Once a row is shorter than the text is tall, labelling every row would
        # overlap into an unreadable smear; skip rows instead.
        step = max(1, math.ceil(fm.height() / row_span_px))

        text_color = QtGui.QColor(*LABEL_TEXT_COLOR)
        chip_color = QtGui.QColor(*LABEL_CHIP_COLOR)

        for i in range(0, n, step):
            abs_ch = self._offset + i
            if abs_ch >= len(self._labels):
                break
            text = self._labels[abs_ch]
            if not text:
                continue
            # top_down puts visible row 0 at the highest world-y.
            world_y = ((n - 1 - i) if self._top_down else i) * z
            y_center = y0 + slope * world_y
            baseline = round(float(y_center + fm.ascent() / 2.0 - fm.descent() / 2.0))
            tw = fm.horizontalAdvance(text)

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(chip_color)
            painter.drawRect(0, baseline - fm.ascent(), tw + 6, fm.height())
            painter.setPen(text_color)
            painter.drawText(3, baseline, text)

        painter.end()


class ScaleBarOverlay(_CanvasOverlay):
    """A labelled vertical calibration bar against the canvas' right edge.

    The owner supplies both the pixel length and the text, because what a
    row-height *means* is a property of the signal's units, which a plotting
    widget does not know. At high channel counts the bar is too short to read;
    it becomes a usable ruler once the user pages down to fewer traces.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._length_px: float = 0.0
        self._text: str = ""

    def set_bar(self, length_px: float, text: str) -> None:
        """Set the bar's pixel length and label.

        A non-positive length or an empty label hides it. Repaints on change.
        """
        length_px = max(0.0, float(length_px))
        if length_px == self._length_px and text == self._text:
            return
        self._length_px, self._text = length_px, text
        self.update()

    def paintEvent(self, event) -> None:
        length = self._length_px
        h, w = self.height(), self.width()
        if not self._text or length < SCALE_BAR_MIN_DRAW_PX or h <= 0 or w <= 0:
            return

        x = w - SCALE_BAR_RIGHT_MARGIN
        y_center = h / 2.0
        y_top = y_center - length / 2.0
        y_bot = y_center + length / 2.0

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        line_color = QtGui.QColor(*LABEL_TEXT_COLOR)
        chip_color = QtGui.QColor(*LABEL_CHIP_COLOR)
        pen = QtGui.QPen(line_color)
        pen.setWidth(2)
        painter.setPen(pen)

        # Vertical bar with serifs at both ends.
        painter.drawLine(int(x), round(float(y_top)), int(x), round(float(y_bot)))
        painter.drawLine(
            int(x - SCALE_BAR_SERIF_HALF), round(float(y_top)), int(x + SCALE_BAR_SERIF_HALF), round(float(y_top))
        )
        painter.drawLine(
            int(x - SCALE_BAR_SERIF_HALF), round(float(y_bot)), int(x + SCALE_BAR_SERIF_HALF), round(float(y_bot))
        )

        # Label, right-aligned to the left of the bar and vertically centred.
        font = QtGui.QFont()
        font.setPixelSize(SCALE_BAR_FONT_PX)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(self._text)
        text_right = x - SCALE_BAR_SERIF_HALF - SCALE_BAR_TEXT_GAP
        text_left = text_right - tw
        baseline = round(float(y_center + fm.ascent() / 2.0 - fm.descent() / 2.0))

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(chip_color)
        painter.drawRect(int(text_left - 3), int(baseline - fm.ascent()), tw + 6, fm.height())
        painter.setPen(line_color)
        painter.drawText(int(text_left), baseline, self._text)
        painter.end()
