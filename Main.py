from interface_principale import MyWindowMenu
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

app = QApplication(sys.argv)

Window = MyWindowMenu()

Window.show()
app.exec()