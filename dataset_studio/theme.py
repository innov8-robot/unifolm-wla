"""Thème sombre, repris de l'identité du cockpit de mpc_any (app/cockpit_sim_ui/theme.py)."""
ACCENT = "#35e0c8"
OK, KO, WARN, MUTED = "#3fd67f", "#f4635e", "#e3b341", "#8b949e"
SERIES = ["#35e0c8", "#e3b341", "#f4635e", "#4a8fe0", "#9acd4e", "#e06a9f", "#8d6fd1", "#e0813a"]
#: couleurs des étiquettes de segments (touches 1 à 9)
TAG_COLORS = ["#4a8fe0", "#e3b341", "#e06a9f", "#9acd4e", "#8d6fd1", "#e0813a", "#35e0c8", "#f4635e", "#c9b28a"]
MONO = "font-family: 'JetBrains Mono','DejaVu Sans Mono',monospace;"

QSS = f"""
* {{ font-family: 'Inter', 'Cantarell', 'DejaVu Sans'; font-size: 13px; }}
QMainWindow, QWidget {{ background: #0e1116; color: #cdd6e0; }}
#Header {{ background: #0a0d12; border-bottom: 1px solid #1c2430; }}
#Title {{ color: #e8eef4; font-size: 15px; font-weight: 600; letter-spacing: 3px; }}
QFrame#Panel {{ background: #11151c; border: 1px solid #1c2430; border-radius: 6px; }}
QLabel#H2 {{ color: {MUTED}; font-size: 11px; letter-spacing: 1.5px; font-weight: 600; }}
QLabel#Cam {{ background: #07090c; border: 1px solid #1c2430; border-radius: 4px; color: {MUTED}; }}
QLabel#Mono {{ {MONO} color: #cdd6e0; }}
QPushButton {{ background: #161c25; border: 1px solid #263041; border-radius: 5px;
               padding: 5px 14px; color: #cdd6e0; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: #4a5462; border-color: #1c2430; }}
QPushButton#Run {{ background: {ACCENT}; color: #06251f; font-weight: 700; border: none; padding: 5px 22px; }}
QPushButton#Run:hover {{ background: #5ff0da; }}
QPushButton#Quiet {{ background: transparent; border: none; color: {MUTED}; padding: 5px 10px; }}
QPushButton#Quiet:hover {{ color: {ACCENT}; }}
QPushButton#Danger {{ color: {WARN}; border-color: #4a3a1c; font-weight: 600; }}
QPushButton#Danger:hover {{ background: #2a2113; color: {KO}; border-color: {KO}; }}
QComboBox, QSpinBox, QLineEdit {{ background: #161c25; border: 1px solid #263041; border-radius: 5px; padding: 4px 8px; }}
QComboBox::drop-down {{ border: none; }}
QTableWidget {{ background: #11151c; border: 1px solid #1c2430; border-radius: 6px; gridline-color: #171d27;
                selection-background-color: #1a2c33; selection-color: #e8eef4; }}
QHeaderView::section {{ background: #0e1116; color: {MUTED}; border: none; border-bottom: 1px solid #1c2430;
                        padding: 5px; font-size: 11px; letter-spacing: 1px; }}
QTableWidget::item {{ padding: 3px 6px; }}
QPlainTextEdit {{ background: #0a0d12; border: 1px solid #1c2430; border-radius: 6px; {MONO} font-size: 12px; }}
QSlider::groove:horizontal {{ height: 6px; background: #1c2430; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QSplitter::handle {{ background: #0e1116; }}
"""
