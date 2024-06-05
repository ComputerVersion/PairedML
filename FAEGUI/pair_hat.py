import sys
import os
import pandas
import csv
from PyQt5.QtWidgets import *
from GUI.Process import Ui_Process
from PyQt5.QtCore import *
from FAEGUI.ProcessConnection import ProcessConnection


# sys.path.append(r"D:/FAE027/FAE-master/FAEGUI/ProcessConnection.py")


class PairedHat(ProcessConnection):
    '''
    加载数据、生成数据对、并把路径加载给Datacontainer
    '''
    def __init__(self,parent = None):
        super(PairedHat, self).__init__(parent)
        self.setupUi(self)

        # self.DatePairGenTrain.clicked.connect(self.LoadTrainingData)
        # self.DatePairGenTest.clicked.connect(self.LoadTestingData)


    # 重写父类的方法LoadTrainingData，使之生成和加载数据对
    def LoadTrainingData(self):
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
                                differ_lines.append(float(mode_label[k]) - float(other_lines[k]))
                            differ_lines = pandas.Series(differ_lines)
                            differ_lines.name = mode_label[0] + '-' + other_lines[0]
                            sub_results = sub_results.join(differ_lines, how='outer')
                    results = results.join(sub_results, how='outer')

            results = results.T
            results[0] = abs(results[0])
            results.columns = list(pandas.read_csv(original_file_name, nrows=0))[1:]
            results.to_csv(file_name, encoding='utf_8_sig')

        try:
            self.training_data_container.Load(file_name)
            self.SetStateButtonBeforeLoading(True)
            self.lineEditTrainingData.setText(file_name)
            self.UpdateDataDescription()
            self.logger.info('Open CSV file ' + file_name + ' succeed.')
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected training data mismatch.')


    # 重写父类的方法LoadTestingData，使之生成和加载数据对
    def LoadTestingData(self):
        dlg = QFileDialog()
        original_file_name, _ = dlg.getOpenFileName(self, 'Open CSV file', filter="csv files (*.csv)")

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
                                differ_lines.append(float(mode_label[k]) - float(other_lines[k]))
                            differ_lines = pandas.Series(differ_lines)
                            differ_lines.name = mode_label[0] + '-' + other_lines[0]
                            sub_results = sub_results.join(differ_lines, how='outer')
                    results = results.join(sub_results, how='outer')

            results = results.T
            results[0] = abs(results[0])
            results.columns = list(pandas.read_csv(original_file_name, nrows=0))[1:]
            results.to_csv(file_name, encoding='utf_8_sig')

        try:
            self.testing_data_container.Load(file_name)
            self.lineEditTestingData.setText(file_name)
            self.UpdateDataDescription()
            self.logger.info('Loading testing data ' + file_name + ' succeed.' )
        except OSError as reason:
            self.logger.log('Open SCV file Error, The reason is ' + str(reason))
            print('出错啦！' + str(reason))
        except ValueError:
            self.logger.error('Open SCV file ' + file_name + ' Failed. because of value error.')
            QMessageBox.information(self, 'Error',
                                    'The selected testing data mismatch.')

    def loadAndDataPairGeneration(self):
        pass

    def dataPairtoObjectProbability(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pair = PairedHat()
    pair.LoadTestingData()