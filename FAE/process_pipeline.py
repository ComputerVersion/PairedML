from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.IndexDict import Index2Dict
from FAE.FeatureAnalysis.Normalizer import NormalizerNone
from FAE.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from FAE.FeatureAnalysis.FeatureSelector import FeatureSelector

# for test
from FAE.FeatureAnalysis.CrossValidation_Pair import CrossValidation5FolderPair as CrossValidationPair


import os
import pickle
import pandas as pd
import csv
import numpy as np
from copy import deepcopy


class OnePipeline:
    def __init__(self, normalizer=None, dimension_reduction=None, feature_selector=None, classifier=None, cross_validation=None):
        self.__normalizer = normalizer
        self.__dimension_reduction = dimension_reduction
        self.__feature_selector = feature_selector
        self.__classifier = classifier
        self.__cv = cross_validation

    def SetNormalizer(self, normalizer):
        self.__normalizer = normalizer
    def GetNormalizer(self):
        return self.__normalizer

    def SetDimensionReduction(self, dimension_reduction):
        self.__dimension_reduction = dimension_reduction
    def GetDimensionReduction(self):
        return self.__dimension_reduction

    def SetFeatureSelector(self, feature_selector):
        self.__feature_selector = feature_selector
    def GetFeatureSelector(self):
        return self.__feature_selector

    def SetClassifier(self, classifier):
        self.__classifier = classifier
    def GetClassifier(self):
        return self.__classifier

    def SetCrossValidation(self, cv):
        self.__cv = cv
    def GetCrossValidatiaon(self):
        return self.__cv

    def SavePipeline(self, feature_number, store_path):
        with open(store_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Normalizer', self.__normalizer.GetName()])
            writer.writerow(['DimensionReduction', self.__dimension_reduction.GetName()])
            writer.writerow(['FeatureSelector', self.__feature_selector.GetName()])
            writer.writerow(['FeatureNumber', feature_number])
            writer.writerow(['Classifier', self.__classifier.GetName()])
            writer.writerow(['CrossValidation', self.__cv.GetName()])

    def LoadPipeline(self, store_path):
        index_2_dict = Index2Dict()
        feature_number = 0
        with open(store_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'Normalizer':
                    self.__normalizer = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'DimensionReduction':
                    self.__dimension_reduction = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'FeatureSelector':
                    self.__feature_selector = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'FeatureNumber':
                    feature_number = int(row[1])
                if row[0] == 'Classifier':
                    self.__classifier = index_2_dict.GetInstantByIndex(row[1])
                if row[0] == 'CrossValidation':
                    self.__cv = index_2_dict.GetInstantByIndex(row[1])
        self.__feature_selector.SetSelectedFeatureNumber(feature_number)

    def GetName(self):
        try:
            return self.__feature_selector[-1].GetName() + '-' + self.__classifier.GetName()
        except:
            return self.__feature_selector.GetName() + '-' + self.__classifier.GetName()

    def GetStoreName(self):
        case_name = self.__normalizer.GetName() + '_' + \
                    self.__dimension_reduction.GetName() + '_' + \
                    self.__feature_selector.GetName() + '_' + \
                    str(self.__feature_selector.GetSelectedFeatureNumber()) + '_' + \
                    self.__classifier.GetName()
        return case_name

    def train_run(self, train_data_container, store_folder, feature_num):
        raw_train_data_container = deepcopy(train_data_container)

        if store_folder:
            if not os.path.exists(store_folder):
                os.mkdir(store_folder)

        if not (self.__cv and self.__classifier):
            print('Give CV method and classifier')
            return

        if self.__normalizer:
            raw_train_data_container = self.__normalizer.Run(raw_train_data_container, store_folder, is_test=False)

        if self.__dimension_reduction:
            raw_train_data_container = self.__dimension_reduction.Run(raw_train_data_container, store_folder)

        if self.__feature_selector:
            self.__feature_selector.SetSelectedFeatureNumber(feature_num)
            raw_train_data_container = self.__feature_selector.Run(raw_train_data_container, store_folder)
            # 将提取的特征写入csv文件
            path = os.path.join(store_folder, "feature_select_info.csv")
            features = raw_train_data_container.GetFeatureName()
            __features = ["selected_features"]
            __features.extend(features)
            with open(path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(__features)

        relative_path = os.path.join(os.path.dirname(__file__), r"HyperParameters\Classifier")
        self.__cv.SetClassifier(self.__classifier)
        # 交叉验证，这一步保存了模型
        self.__cv.Run(raw_train_data_container, relative_path, store_folder)

    def test_run(self, data_container, store_folder, feature_num):
        raw_data_container = deepcopy(data_container)

        if self.__normalizer:
            raw_data_container = self.__normalizer.Run(raw_data_container, store_folder, is_test=True)

        if self.__dimension_reduction:
            raw_data_container = self.__dimension_reduction.Transform(raw_data_container, store_folder)

        if self.__feature_selector:
            self.__feature_selector.SetSelectedFeatureNumber(feature_num)
            select_feature_df = pd.read_csv(os.path.join(store_folder, "feature_select_info.csv"), index_col=0)
            select_feature_name = list(select_feature_df.columns)
            raw_data_container = self.__feature_selector.SelectFeatureByName(raw_data_container, select_feature_name)

        # 加载模型
        self.__classifier.Load(store_folder)
        # pred = self.__classifier.Predict(raw_data_container.GetArray())
        if raw_data_container.GetArray().size > 0:
            test_data = raw_data_container.GetArray()
            test_label = raw_data_container.GetLabel()
            test_case_name = raw_data_container.GetCaseName()
            # test_pred = self.classifier.Predict(test_data)
            test_pred = self.__classifier.Predict(test_data)

            print("test_pred",test_pred)

            # 修改4，共4处，为了增加Pair功能，更新所有测试集的概率、标签、名字
            all_test_case_name = pd.DataFrame(test_case_name)
            all_test_pred = pd.DataFrame(test_pred)
            all_test_label = pd.DataFrame(test_label)
            all_test_prob_label_name = pd.concat([all_test_case_name, all_test_label, all_test_pred], axis=1)
            all_test_prob_label_name.columns = ['name', 'label', 'prob']
            all_test_prob_label_name.set_index(["name"], inplace=True)

            # train_sortedProbabilitybyname = self.pariedtoObjectProbability(all_test_prob_label_name)
            cvp = CrossValidationPair()
            train_sortedProbabilitybyname = cvp.pariedtoObjectProbability(all_test_prob_label_name)

            print("train_sortedProbabilitybyname", train_sortedProbabilitybyname)

            test_pred = train_sortedProbabilitybyname['probability']

            print("test_pred",test_pred)

        return test_pred

if __name__ == '__main__':
    data_container = DataContainer()
    file_path = os.path.abspath(r'..\..\Example\numeric_feature.csv')
    print(file_path)
    data_container.Load(file_path)

    # temp = OnePipeline(normalizer=NormalizerZeroCenterAndUnit(), feature_selector=FeatureSelectPipeline([RemoveCosSimilarityFeatures(), FeatureSelectByANOVA(10)]),
    #                    classifier=SVM(), cross_validation=CrossValidation('5-folder'))
    # temp.Run(data_container, store_folder=r'..\..\Example\one_pipeline')