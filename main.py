import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from GUI import app_window

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = app_window.MainWindow()
    main_window.show()
    sys.exit(app.exec_())
