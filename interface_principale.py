from Menu_Interface import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtGui import QIcon


class MyWindowMenu(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI for Computer Vision Project")
        self.setGeometry(500, 150, 1000, 800)
        self.setWindowIcon(QIcon("images/desktop.png"))

        # Initialize the UI
        self.setupUi(self)

        # Initialize custom behavior
        self.initUI()

    def initUI(self):
        # Connect buttons to the switching methods
        self.pushButton_4.clicked.connect(self.switch_to_part1Page)  # For "Partie1"
        self.pushButton_5.clicked.connect(self.switch_to_part2Page)  # For "Partie2"
        self.pushButton_6.clicked.connect(self.switch_to_part3Page)  # For "Partie3"
        self.pushButton_11.clicked.connect(self.switch_to_part4Page)  # For "Partie3"

    def switch_to_part1Page(self):
        """Switch to the page corresponding to Part 1."""
        self.stackedWidget.setCurrentIndex(1)  # Index 0 for Part 1

    def switch_to_part2Page(self):
        """Switch to the page corresponding to Part 2."""
        self.stackedWidget.setCurrentIndex(2)  # Index 1 for Part 2

    def switch_to_part3Page(self):
        """Switch to the page corresponding to Part 3."""
        self.stackedWidget.setCurrentIndex(3)  # Index 2 for Part 3

    def switch_to_part4Page(self):
        """Switch to the page corresponding to Part 3."""
        self.stackedWidget.setCurrentIndex(4)  # Index 2 for Part 3


