# admin_gui/qt_compat.py
"""PyQt6/PyQt5 호환 레이어. 이 패키지의 다른 모든 모듈은 PyQt를 직접 import하지 않고
여기서 가져다 쓴다 — 버전 분기 코드를 한 군데로 모으기 위함."""
import os

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

try:
    from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPen, QPixmap
    from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
                                 QFormLayout, QFrame, QGraphicsDropShadowEffect, QGridLayout,
                                 QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                                 QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                                 QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                                 QWidget)
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
                                 QFormLayout, QFrame, QGraphicsDropShadowEffect, QGridLayout,
                                 QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                                 QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                                 QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                                 QWidget)
    PYQT_VERSION = 5

__all__ = [
    'Qt', 'QTimer', 'QRectF', 'pyqtSignal',
    'QColor', 'QFont', 'QFontDatabase', 'QIcon', 'QImage', 'QPainter', 'QPen', 'QPixmap',
    'QAbstractItemView', 'QApplication', 'QComboBox', 'QDialog', 'QFormLayout', 'QFrame',
    'QGraphicsDropShadowEffect', 'QGridLayout', 'QHBoxLayout', 'QHeaderView', 'QLabel',
    'QLineEdit', 'QListWidget', 'QListWidgetItem', 'QMainWindow',
    'QMessageBox', 'QPushButton', 'QScrollArea', 'QTableWidget', 'QTableWidgetItem', 'QTabWidget',
    'QVBoxLayout', 'QWidget',
    'PYQT_VERSION',
]
