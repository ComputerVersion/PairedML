import os
import pandas
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from FAE.FeatureAnalysis.Normalizer import *
from FAE.FeatureAnalysis.DimensionReduction import *
from FAE.FeatureAnalysis.FeatureSelector import *
from FAE.FeatureAnalysis.Classifier import *
from FAE.process_pipeline import OnePipeline
from FAE.FeatureAnalysis.CrossValidation import *


# from FAEGUI.ProcessConnection import *

from FAE.FeatureAnalysis.CrossValidation_Pair import *

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(1000, 500)
        self.main_layout = QGridLayout()
        self.setStyleSheet("background-color:white")

        self.label_title = QLabel()
        self.frame_all_fun = QFrame()
        self.frame_res = QFrame()

        self.main_layout.addWidget(self.label_title, 0, 0, 1, 10)
        self.main_layout.addWidget(self.frame_all_fun, 1, 0, 15, 7)
        self.main_layout.addWidget(self.frame_res, 1, 7, 15, 3)

        self.setLayout(self.main_layout)

        self.font_title = QFont("Times New Roman")
        self.font_title.setPointSize(15)  # 设置字体大小

        self.font_content = QFont("等线")
        self.font_content.setPointSize(12)  # 设置字体大小

        self.logger = eclog(os.path.split(__file__)[-1]).GetLogger()
        self.data_container = DataContainer()
        self.store_folder = os.getcwd() # 当前文件所在的目录
        self.moudle_folder = None

        self.normalizer = None
        self.dimension_reduction = None
        self.feature_selector = None
        self.feature_num = None
        self.classifier = None
        self.cross_validation = None
        self.pipeline = OnePipeline()

        self.res = None

        self.init_title()
        self.init_fun_frame()
        self.init_res_frame()

    def init_title(self):
        self.label_title.setText("Pairwise ML Intelligent Diagnosis System for Esophageal Cancer")
        self.label_title.setStyleSheet("background-color:rgb(255, 240, 157); border: 1px solid black;")
        # 设置内容居中对齐
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setFont(self.font_title)

    def init_fun_frame(self):
        self.frame_data_Load = FunFrame("DataLoad", self.font_content)
        self.frame_normalization = FunFrame("Normalization", self.font_content)
        self.frame_dimred = FunFrame("DimRed.", self.font_content)
        self.frame_feasel = FunFrame("FeaSel.", self.font_content)
        self.frame_classifier = FunFrame("Classifier", self.font_content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.frame_data_Load, 8)
        layout.addWidget(self.frame_normalization, 11)
        layout.addWidget(self.frame_dimred, 7)
        layout.addWidget(self.frame_feasel, 8)
        layout.addWidget(self.frame_classifier, 16)

        # 添加分割线
        count = layout.count()
        for i in range(count):
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.insertWidget(count - i, line)

        self.frame_all_fun.setLayout(layout)

        # 创建按钮并分组
        self.button_csv = QPushButton("Gen and Load Train Data Pair")
        self.button_load_image = QPushButton("Gen and Load Test Data Pair")
        # self.button_mask_load = QPushButton("MaskLoad")

        self.button_norm2unit = QRadioButton("Norm2Unit")
        self.button_norm2cen = QRadioButton("Norm2-0-Cen")
        self.button_norm2unit_cen = QRadioButton("Norm2Unit-0-Cen")
        self.button_group_normalization = QButtonGroup()
        self.button_group_normalization.addButton(self.button_norm2cen)
        self.button_group_normalization.addButton(self.button_norm2unit)
        self.button_group_normalization.addButton(self.button_norm2unit_cen)

        self.button_pca = QRadioButton("PCA")
        self.button_pcc = QRadioButton("PCC")
        self.button_group_dimred = QButtonGroup()
        self.button_group_dimred.addButton(self.button_pca)
        self.button_group_dimred.addButton(self.button_pcc)

        self.button_anova = QRadioButton("ANOVA")
        self.button_rfe = QRadioButton("RFE")
        self.button_relief = QRadioButton("Relief")
        self.button_group_feasel = QButtonGroup()
        self.button_group_feasel.addButton(self.button_anova)
        self.button_group_feasel.addButton(self.button_rfe)
        self.button_group_feasel.addButton(self.button_relief)

        self.button_svm = QRadioButton("SVM")
        self.button_ae = QRadioButton("AE")
        self.button_lr = QRadioButton("LR")
        self.button_nb = QRadioButton("NB")
        self.button_rf = QRadioButton("RF")
        self.button_adaboost = QRadioButton("Adaboost")
        self.button_group_classifier = QButtonGroup()
        self.button_group_classifier.addButton(self.button_svm)
        self.button_group_classifier.addButton(self.button_ae)
        self.button_group_classifier.addButton(self.button_lr)
        self.button_group_classifier.addButton(self.button_nb)
        self.button_group_classifier.addButton(self.button_rf)
        self.button_group_classifier.addButton(self.button_adaboost)

        # 设置DataLoad窗口
        def init_dataload():
            # frame = QFrame()
            frame_layout = QHBoxLayout()
            frame_layout.setContentsMargins(10, 4, 10, 4)
            frame_layout.addWidget(self.button_csv, 7)
            frame_layout.addStretch(2)

            frame_layout.addWidget(self.button_load_image, 7)
            frame_layout.addStretch(5)

            # frame.setLayout(frame_layout)
            # frame.setStyleSheet("background-color:rgb(255, 224, 187);\n"
            #                     "border-radius:1px")

            self.button_csv.setStyleSheet("QPushButton{"
                                          "background-color:rgb(255, 182, 108);\n"
                                          "border-width:1px;\n"
                                          "border-style:solid;\n"
                                          "border-color:black;\n"
                                          "border-radius:6px;\n"
                                          "font: 12pt \"等线\";}"
                                          "QPushButton::hover{"
                                          "background-color:rgb(255,160,90);}"
                                          "QPushButton::pressed{"
                                          "background-color:rgb(255, 140, 70);}"
                                          )
            self.button_load_image.setStyleSheet("QPushButton{"
                                                 "background-color:rgb(255, 182, 108);\n"
                                                 "border-width:1px;\n"
                                                 "border-style:solid;\n"
                                                 "border-color:black;\n"
                                                 "border-radius:6px;\n"
                                                 "font: 12pt \"等线\";}"
                                                 "QPushButton::hover{"
                                                 "background-color:rgb(255,160,90);}"
                                                 "QPushButton::pressed{"
                                                 "background-color:rgb(255, 140, 70);}"
                                                 )


            self.button_csv.setMaximumHeight(100)
            self.button_load_image.setMaximumHeight(100)
            # self.button_mask_load.setMaximumHeight(100)

            self.button_load_image.clicked.connect(self.LoadTestDataPair)             # 生成并加载测试集数据对    修改 2
            self.button_csv.clicked.connect(self.LoarTrainDataPair)                     # 生成并加载训练集数据对   修改 1

            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(1, 1, 5, 1)
            main_layout.setSpacing(15)
            main_layout.addWidget(self.button_csv, 1)
            # main_layout.addWidget(frame, 2)
            main_layout.addWidget(self.button_load_image, 1)


            self.frame_data_Load.frame_buttons.setLayout(main_layout)



        # 设置Normalization窗口
        def init_normalization():
            self.button_norm2unit.setStyleSheet("QRadioButton{"
                                                "background-color:rgb(255, 182, 108);"
                                                "border-width:1px;"
                                                "border-style:solid;"
                                                "border-color:black;"
                                                "border-radius:6px;"
                                                "font: 12pt \"等线\";"
                                                "spacing: 10px;}"
                                                "QRadioButton::indicator{"
                                                "width: 20px;"
                                                "height: 20px;"
                                                "padding-left:10px}")
            self.button_norm2cen.setStyleSheet("QRadioButton{"
                                                "background-color:rgb(255, 182, 108);"
                                                "border-width:1px;"
                                                "border-style:solid;"
                                                "border-color:black;"
                                                "border-radius:6px;"
                                                "font: 12pt \"等线\";"
                                                "spacing: 10px;}"
                                                "QRadioButton::indicator{"
                                                "width: 20px;"
                                                "height: 20px;"
                                                "padding-left:10px}")
            self.button_norm2unit_cen.setStyleSheet("QRadioButton{"
                                                    "background-color:rgb(255, 182, 108);"
                                                    "border-width:1px;"
                                                    "border-style:solid;"
                                                    "border-color:black;"
                                                    "border-radius:6px;"
                                                    "font: 12pt \"等线\";"
                                                    "spacing: 10px;}"
                                                    "QRadioButton::indicator{"
                                                    "width: 20px;"
                                                    "height: 20px;"
                                                    "padding-left:10px}")
            self.button_norm2unit_cen.setMaximumHeight(100)
            self.button_norm2unit.setMaximumHeight(100)
            self.button_norm2cen.setMaximumHeight(100)
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(1, 1, 5, 1)
            main_layout.setSpacing(15)
            main_layout.addWidget(self.button_norm2unit)
            main_layout.addWidget(self.button_norm2cen)
            main_layout.addWidget(self.button_norm2unit_cen)
            self.frame_normalization.frame_buttons.setLayout(main_layout)

        # 设置DimRed.窗口
        def init_dimred():
            self.button_pcc.setStyleSheet("QRadioButton{"
                                          "background-color:rgb(255, 182, 108);"
                                          "border-width:1px;"
                                          "border-style:solid;"
                                          "border-color:black;"
                                          "border-radius:6px;"
                                          "font: 12pt \"等线\";"
                                          "spacing: 90px;}"
                                          "QRadioButton::indicator{"
                                          "width: 20px;"
                                          "height: 20px;"
                                          "padding-left:10px}")
            self.button_pca.setStyleSheet("QRadioButton{"
                                          "background-color:rgb(255, 182, 108);"
                                          "border-width:1px;"
                                          "border-style:solid;"
                                          "border-color:black;"
                                          "border-radius:6px;"
                                          "font: 12pt \"等线\";"
                                          "spacing: 90px;}"
                                          "QRadioButton::indicator{"
                                          "width: 20px;"
                                          "height: 20px;"
                                          "padding-left:10px}")
            self.button_pcc.setMaximumHeight(100)
            self.button_pca.setMaximumHeight(100)
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(1, 1, 5, 1)
            main_layout.setSpacing(30)
            main_layout.addWidget(self.button_pca)
            main_layout.addWidget(self.button_pcc)
            self.frame_dimred.frame_buttons.setLayout(main_layout)

        # 设置FeaSel.窗口
        def init_feasel():
            self.button_anova.setStyleSheet("QRadioButton{"
                                            "background-color:white;"
                                            "border-width:1px;"
                                            "border-style:solid;"
                                            "border-color:black;"
                                            "border-radius:6px;"
                                            "font: 12pt \"等线\";"
                                            "spacing: 20px;}"
                                            "QRadioButton::indicator{"
                                            "width: 20px;"
                                            "height: 20px;"
                                            "padding-left:10px}")
            self.button_rfe.setStyleSheet("QRadioButton{"
                                          "background-color:white;"
                                          "border-width:1px;"
                                          "border-style:solid;"
                                          "border-color:black;"
                                          "border-radius:6px;"
                                          "font: 12pt \"等线\";"
                                          "spacing: 20px;}"
                                          "QRadioButton::indicator{"
                                          "width: 20px;"
                                          "height: 20px;"
                                          "padding-left:10px}")
            self.button_relief.setStyleSheet("QRadioButton{"
                                             "background-color:white;"
                                             "border-width:1px;"
                                             "border-style:solid;"
                                             "border-color:black;"
                                             "border-radius:6px;"
                                             "font: 12pt \"等线\";"
                                             "spacing: 20px;}"
                                             "QRadioButton::indicator{"
                                             "width: 20px;"
                                             "height: 20px;"
                                             "padding-left:10px}")
            self.button_anova.setMaximumHeight(100)
            self.button_rfe.setMaximumHeight(100)
            self.button_relief.setMaximumHeight(100)
            frame = QFrame()
            frame_layout = QHBoxLayout()
            frame_layout.setContentsMargins(4, 1, 4, 1)
            frame_layout.setSpacing(80)
            frame_layout.addWidget(self.button_anova)
            frame_layout.addWidget(self.button_rfe)
            frame_layout.addWidget(self.button_relief)
            frame.setLayout(frame_layout)
            frame.setStyleSheet("background-color:rgb(223, 178, 255);"
                                "border-radius:6px;"
                                "border-width:1px;"
                                "border-style:solid;"
                                "border-color:black;"
                                "border-radius:6px;")
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(1, 1, 5, 1)
            main_layout.addWidget(frame)
            self.frame_feasel.frame_buttons.setLayout(main_layout)

        # 设置Classifier窗口
        def init_classifier():
            self.button_svm.setStyleSheet("QRadioButton{"
                                          "background-color:white;"
                                          "border-width:1px;"
                                          "border-style:solid;"
                                          "border-color:black;"
                                          "border-radius:6px;"
                                          "font: 12pt \"等线\";"
                                          "spacing: 40px;}"
                                          "QRadioButton::indicator{"
                                          "width: 20px;"
                                          "height: 20px;"
                                          "padding-left:10px}")
            self.button_ae.setStyleSheet("QRadioButton{"
                                         "background-color:white;"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";"
                                         "spacing: 40px;}"
                                         "QRadioButton::indicator{"
                                         "width: 20px;"
                                         "height: 20px;"
                                         "padding-left:10px}")
            self.button_lr.setStyleSheet("QRadioButton{"
                                         "background-color:white;"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";"
                                         "spacing: 40px;}"
                                         "QRadioButton::indicator{"
                                         "width: 20px;"
                                         "height: 20px;"
                                         "padding-left:10px}")
            self.button_nb.setStyleSheet("QRadioButton{"
                                         "background-color:white;"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";"
                                         "spacing: 40px;}"
                                         "QRadioButton::indicator{"
                                         "width: 20px;"
                                         "height: 20px;"
                                         "padding-left:10px}")
            self.button_rf.setStyleSheet("QRadioButton{"
                                         "background-color:white;"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";"
                                         "spacing: 40px;}"
                                         "QRadioButton::indicator{"
                                         "width: 20px;"
                                         "height: 20px;"
                                         "padding-left:10px}")
            self.button_adaboost.setStyleSheet("QRadioButton{"
                                               "background-color:white;"
                                               "border-width:1px;"
                                               "border-style:solid;"
                                               "border-color:black;"
                                               "border-radius:6px;"
                                               "font: 12pt \"等线\";"
                                               "spacing: 40px;}"
                                               "QRadioButton::indicator{"
                                               "width: 20px;"
                                               "height: 20px;"
                                               "padding-left:10px}")
            self.button_svm.setMaximumHeight(100)
            self.button_ae.setMaximumHeight(100)
            self.button_adaboost.setMaximumHeight(100)
            self.button_lr.setMaximumHeight(100)
            self.button_nb.setMaximumHeight(100)
            self.button_rf.setMaximumHeight(100)
            frame = QFrame()
            frame_layout = QGridLayout()
            frame_layout.setContentsMargins(4, 4, 4, 4)
            frame_layout.addWidget(self.button_svm, 0, 0, 1, 1)
            frame_layout.addWidget(self.button_ae, 0, 1, 1, 1)
            frame_layout.addWidget(self.button_lr, 0, 2, 1, 1)
            frame_layout.addWidget(self.button_nb, 1, 0, 1, 1)
            frame_layout.addWidget(self.button_rf, 1, 1, 1, 1)
            frame_layout.addWidget(self.button_adaboost, 1, 2, 1, 1)
            frame.setLayout(frame_layout)
            frame.setStyleSheet("background-color:rgb(53, 255, 181);"
                                "border-radius:6px;"
                                "border-width:1px;"
                                "border-style:solid;"
                                "border-color:black;"
                                "border-radius:6px;")
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(1, 1, 5, 1)
            main_layout.addWidget(frame)
            self.frame_classifier.frame_buttons.setLayout(main_layout)

        init_dataload()
        init_normalization()
        init_dimred()
        init_feasel()
        init_classifier()

    def init_res_frame(self):
        self.frame_res.setStyleSheet("QFrame{"
                                     "background-color:rgb(255, 203, 192);"
                                     "border-radius:6px}"
                                     "QLineEdit{"
                                     "border-style:solid;"
                                     "border-color:black;"
                                     "border-width:1px})"
                                     )
        self.label_local_regional_control = QLabel()
        self.line_edit_local_regional_control = QLineEdit()
        self.line_edit_local_regional_control.setReadOnly(True)
        self.label_local_regional_control.setText("Local Regional Control\n(0 or 1)")
        self.label_local_regional_control.setAlignment(QtCore.Qt.AlignCenter)
        self.label_local_regional_control.setFont(self.font_content)

        self.label_local_control = QLabel()
        self.line_edit_local_control = QLineEdit()
        self.line_edit_local_control.setReadOnly(True)
        self.label_local_control.setText("Local Control\n(0 or 1)")
        self.label_local_control.setAlignment(QtCore.Qt.AlignCenter)
        self.label_local_control.setFont(self.font_content)

        self.label_treatment_response = QLabel()
        self.line_edit_treatment_response = QLineEdit()
        self.line_edit_treatment_response.setReadOnly(True)
        self.label_treatment_response.setText("Treatment response\n(0 or 1)")
        self.label_treatment_response.setAlignment(QtCore.Qt.AlignCenter)
        self.label_treatment_response.setFont(self.font_content)

        self.label_survival = QLabel()
        self.line_edit_survival = QLineEdit()
        self.line_edit_survival.setReadOnly(True)
        self.label_survival.setText("Survival\n(0 or 1)")
        self.label_survival.setAlignment(QtCore.Qt.AlignCenter)
        self.label_survival.setFont(self.font_content)

        self.button_submit = QPushButton("Submit")
        self.button_submit.clicked.connect(lambda: self.run(True))
        self.button_submit.setStyleSheet("QPushButton{"
                                         "background-color:rgb(255, 85, 127);"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";}"
                                         "QPushButton::hover{"
                                         "background-color:rgb(240,100,150);}"
                                         "QPushButton::pressed{"
                                         "background-color:rgb(230, 120, 170);}")
        self.button_train = QPushButton("Train")
        self.button_train.clicked.connect(lambda: self.run(False))                               # concerned with the training process with data pair
        self.button_train.setStyleSheet("QPushButton{"
                                         "background-color:rgb(255, 85, 127);"
                                         "border-width:1px;"
                                         "border-style:solid;"
                                         "border-color:black;"
                                         "border-radius:6px;"
                                         "font: 12pt \"等线\";}"
                                         "QPushButton::hover{"
                                         "background-color:rgb(240,100,150);}"
                                         "QPushButton::pressed{"
                                         "background-color:rgb(230, 120, 170);}")
        self.button_submit.setMaximumWidth(200)
        self.button_submit.setMinimumHeight(30)
        self.button_train.setMaximumWidth(200)
        self.button_train.setMinimumHeight(30)
        hbox = QHBoxLayout()
        hbox.addStretch(2)
        hbox.addWidget(self.button_train, 5)
        hbox.addWidget(self.button_submit, 5)
        hbox.addStretch(2)

        main_layout = QVBoxLayout()
        main_layout.addStretch(1)
        main_layout.addWidget(self.label_local_regional_control, 2)
        main_layout.addWidget(self.line_edit_local_regional_control, 1)
        main_layout.addWidget(self.label_local_control, 2)
        main_layout.addWidget(self.line_edit_local_control, 1)
        main_layout.addWidget(self.label_treatment_response, 2)
        main_layout.addWidget(self.line_edit_treatment_response, 1)
        main_layout.addWidget(self.label_survival, 2)
        main_layout.addWidget(self.line_edit_survival, 1)
        main_layout.addStretch(1)
        main_layout.addLayout(hbox, 1)

        self.frame_res.setLayout(main_layout)

    def LoarTrainDataPair(self):
        # ld = ProcessConnection()
        # ld.LoadTrainDataPair()
        dlg = QFileDialog()
        original_file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=r'C:\MyCode\FAE\Example', filter="csv files (*.csv)")
        # 生成新路径，即生成保存文件路径file_name
        filen, extension = os.path.splitext(original_file_name)
        file_name = os.path.join(filen + "_" + "paired" + extension)
        results = pandas.DataFrame()
        label_number = 0
        with open(original_file_name, 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            # go through all lines
            for lines in reader:
                # do a mark for the line with label
                label_number = label_number + 1

                # find a  line with label
                if 'label' in lines[0]:
                    sub_results = pandas.DataFrame()
                    mode_label = lines
                    sub_label_number = 0

                    with open(original_file_name, 'r', encoding="utf-8") as again_csvfile:
                        reader_again = csv.reader(again_csvfile)
                        next(reader_again)
                        for other_lines in reader_again:
                            # mark other lines with label
                            sub_label_number = sub_label_number + 1

                            # skip if encounter the same case
                            if sub_label_number == label_number or 'label' in other_lines[0]:
                                continue

                            # calculate the difference
                            differ_lines = []
                            for k in range(1, len(other_lines)):
                                print(len(other_lines))

                                # FOR TEST
                                print(float(mode_label[k]))
                                print(float(other_lines[k]))

                                differ_lines.append(float(mode_label[k]) - float(other_lines[k]))
                            differ_lines = pandas.Series(differ_lines)
                            differ_lines.name = mode_label[0] + '-' + other_lines[0]
                            sub_results = sub_results.join(differ_lines, how='outer')
                    results = results.join(sub_results, how='outer')

            results = results.T
            results[0] = abs(results[0])
            results.columns = list(pandas.read_csv(original_file_name, nrows=0))[1:]
            results.to_csv(file_name, encoding='utf-8')

            self.store_folder = os.path.dirname(file_name)

            # test
            # print("file_name: ", file_name)

        try:
            # self.training_data_container.Load(file_name)
            self.data_container.Load(file_name)
            # self.SetStateButtonBeforeLoading(True)
            # self.lineEditTrainingData.setText(file_name)
            # self.UpdateDataDescription()
            self.logger.info('Open CSV file ' + file_name + ' succeed.')
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected training data mismatch.')


    def LoadTestDataPair(self):
        # ld = ProcessConnection()
        # ld.LoadTestDataPair()
        dlg = QFileDialog()
        original_file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")

        # 生成新路径，即生成保存文件路径file_name
        filen, extension = os.path.splitext(original_file_name)
        file_name = os.path.join(filen + "_" + "paired" + extension)

        self.store_folder = os.path.dirname(file_name)

        results = pandas.DataFrame()
        label_number = 0
        with open(original_file_name, 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            # go through all lines
            for lines in reader:
                # do a mark for the line with label
                label_number = label_number + 1

                # find a  line with label
                if 'label' in lines[0]:
                    sub_results = pandas.DataFrame()
                    mode_label = lines
                    sub_label_number = 0

                    with open(original_file_name, 'r', encoding="utf-8") as again_csvfile:
                        reader_again = csv.reader(again_csvfile)
                        next(reader_again)
                        for other_lines in reader_again:
                            # mark other lines with label
                            sub_label_number = sub_label_number + 1

                            # skip if encounter the same case
                            if sub_label_number == label_number or 'label' in other_lines[0]:
                                continue

                            # calculate the difference
                            differ_lines = []
                            for k in range(1, len(other_lines)):
                                print(len(other_lines))
                                differ_lines.append(float(mode_label[k]) - float(other_lines[k]))
                            differ_lines = pandas.Series(differ_lines)
                            differ_lines.name = mode_label[0] + '-' + other_lines[0]
                            sub_results = sub_results.join(differ_lines, how='outer')
                    results = results.join(sub_results, how='outer')

            results = results.T
            results[0] = abs(results[0])
            results.columns = list(pandas.read_csv(original_file_name, nrows=0))[1:]
            results.to_csv(file_name, encoding='utf-8')

        try:
            # self.testing_data_container.Load(file_name)
            self.data_container.Load(file_name)
            # self.lineEditTestingData.setText(file_name)
            # self.UpdateDataDescription()
            self.logger.info('Loading testing data ' + file_name + ' succeed.' )
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected testing data mismatch.')

    def LoadData(self):
        dlg = QFileDialog()
        file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', directory=self.store_folder, filter="csv files (*.csv)")
        self.store_folder = os.path.dirname(file_name)
        # for test
        # file_name: C:/Users/Gerald/Desktop/Demo/data/train_numeric_feature.csv
        try:
            self.data_container.Load(file_name)
            # self.lineEditTrainingData.setText(file_name)
            # self.UpdateDataDescription()
            self.logger.info('Open CSV file ' + file_name + ' succeed.')

        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected training data mismatch.')

    def run(self, is_test):
        if self.data_container is None:
            self.line_edit_local_regional_control.setText("")
            return

        self.line_edit_local_regional_control.setText("")
        if self.button_norm2unit_cen.isChecked():
            self.normalizer = NormalizerZeroCenterAndUnit
        # elif self.button_norm2unit.isChecked():
        #     self.normalizer = NormalizerUnit
        # elif self.button_norm2cen.isChecked():
        #     self.normalizer = NormalizerZeroCenter
        else:
            return

        # if self.button_pca.isChecked():
        #     self.dimension_reduction = DimensionReductionByPCA()
        if self.button_pcc.isChecked():
            self.dimension_reduction = DimensionReductionByPCC()
        else:
            return

        if self.button_anova.isChecked():
            self.feature_selector = FeatureSelectByANOVA()
        else:
            return
        # elif self.button_rfe.isChecked():
        #     self.feature_selector = FeatureSelectByRFE()
        # elif self.button_relief.isChecked():
        #     self.feature_selector = FeatureSelectByRelief()

        if self.button_nb.isChecked():
            self.classifier = NaiveBayes()
        else:
            return
        # if self.button_rf.isChecked():
        #     self.classifier = RandomForest()
        # elif self.button_svm.isChecked():
        #     self.classifier = SVM()
        # elif self.button_ae.isChecked():
        #     self.classifier = AE()
        # elif self.button_lr.isChecked():
        #     self.classifier = LR()
        # elif self.button_nb.isChecked():
        #     self.classifier = NaiveBayes()
        # elif self.button_adaboost.isChecked():
        #     self.classifier = AdaBoost()

        # self.cross_validation = CrossValidation5Folder()                                    # For data-pair   5折交叉验证
        self.cross_validation = CrossValidation5FolderPair()                                 # For data-pair   5折交叉验证

        self.pipeline.SetNormalizer(self.normalizer)
        self.pipeline.SetDimensionReduction(self.dimension_reduction)
        self.pipeline.SetFeatureSelector(self.feature_selector)
        self.pipeline.SetClassifier(self.classifier)
        self.pipeline.SetCrossValidation(self.cross_validation)

        self.feature_num = 12
        if is_test:
            if self.moudle_folder is None:
                self.moudle_folder = QFileDialog.getExistingDirectory(self, "选择存储文件夹")
            self.res = self.pipeline.test_run(self.data_container, self.moudle_folder, self.feature_num)
            result = self.res[0]
            print(result)
            if result <= 0.5:
                self.line_edit_local_regional_control.setText("0")
            else:
                self.line_edit_local_regional_control.setText("1")

        else:
            self.moudle_folder = QFileDialog.getExistingDirectory(self, "选择存储文件夹")
            # print(self.store_folder)
            self.pipeline.train_run(self.data_container, self.moudle_folder, self.feature_num)
            print("训练完成")

class FunFrame(QFrame):
    def __init__(self, title, font, ):
        super(FunFrame, self).__init__()
        self.label_title = QLabel()
        self.label_title.setText(title)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setFont(font)

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_buttons = QFrame()
        self.main_layout.addWidget(self.label_title, 1)
        self.main_layout.addWidget(self.frame_buttons, 5)

        self.setLayout(self.main_layout)
        self.setStyleSheet("FunFrame {border: 1px solid rgb(211, 211, 211);}")

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())