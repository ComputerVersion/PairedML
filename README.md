# Diagnostic Platform User Guide
In this project, we developed two interface tools, one based on the conventional modeling process for machine models, and the other based on the modeling process of machine learning models using paired samples.
## Libraries Required
```
- imbalanced-learn
- matplotlib (seaborn)
- numpy
- pandas
- pymrmr
- PyQt5
- pyradiomics
- reportlab
- scikit-learn
- scipy
```
## File Description
- **FAE**. Core code of the entire pipeline processing
- **GUI**.Implementation of the graphical interface
- **Test_example**.Test data example
- **Train_example**.Training data examples and intermediate results, including the models used
- **Utility**.Some small components, including the implementation of logging


## Usage of the Machine Model Based on the Conventional Modeling Process
1. Run the ***main.py*** file to open the interface.

2. Click the ***Radiomics.CSV*** button to read training data or test data.

3. Select pipeline tools (currently only supports ***Norm2Unit-0-Cen -> PCC -> ANOVA -> NB***).

4. Click the ***Train*** button to train data, click the ***Submit*** button to test data, the result output is 0 or 1.

***Note: When training, you need to choose the path to save the model, the path for testing will default to the path saved during training. If not trained, after clicking the ***Submit*** button, you need to choose the path of the model from the previous training step.***

## Modeling Process for Machine Learning Models Based on Paired Samples
1. Run the `main_pair.py` file to open the interface.

2. Click the `Gen and Load Train Data Pair` button to generate and load training data pairs; click the `Gen and Load Test Data Pair` button to generate and load test data pairs.

3. Select pipeline tools (currently only supports `Norm2Unit-0-Cen -> PCC -> ANOVA -> NB`).

4. Click the `Train` button to train data, click the `Submit` button to test data, with the results outputting 0 or 1.

   **Note: When training, you need to select the path to save the model, and the path for testing will default to the path saved during training. If no training has been conducted, after clicking the `Submit` button, you will need to select the path of the model from the previous training step.**

## Data Format
Please refer to the `train_numeric_feature.csv` file in the `Pair_Test_example_Data` directory for the training data format, and refer to the `test_numeric_feature0.csv` file or the `test_numeric_feature1.csv` file for the test data format. **Please ensure that the test data and training data formats are the beyond.**

## Workflow Description
Before processing, the data must first be loaded. The implementation of the data container is located in the `Container` folder under the `FAE` directory.
The entire processing flow is implemented in the `process_pipeline.py` file in the `FAE` directory, including the training and testing processes, which are algorithmically similar, but differ in that the training process yields intermediate results and saves them, whereas the testing process loads these intermediate results and uses them directly. The implementation steps of the algorithm include normalization, dimension reduction, feature extraction, classification, and cross-validation, with each solver used in these steps located in the `FeatureAnalysis` folder.

## If this work is helpful to you, please cite the following paper. Thank you!
Zhang AD, Shi QL, Zhang HT, Duan WH, Li Y, Ruan L, Han YF, Liu ZK, Li HF, Xiao JS, Shi GF, Wan X, Wang RZ. Pairwise machine learning-based automatic diagnostic platform utilizing CT images and clinical information for predicting radiotherapy locoregional recurrence in elderly esophageal cancer patients. Abdom Radiol (NY). 2024 Jun 4. doi: 10.1007/s00261-024-04377-7. Epub ahead of print. PMID: 38831075.
