"""Qt fixture for widget tests.

The overlays are plain ``QWidget`` painters -- no GPU, no fastplotlib -- so they
can be exercised on the offscreen platform plugin. That keeps the geometry maths
(which is where the bugs live) under test on a headless runner, without needing
a rendering backend.
"""

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A QApplication on the offscreen platform, or skip."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            app = QtWidgets.QApplication([])
        except Exception as exc:  # noqa: BLE001 - any platform failure means skip
            pytest.skip(f"cannot start QApplication: {exc}")
    return app
