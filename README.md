## HEART DISEASE PREDICTION USING MACHINE LEARNING METHODOLOGIES:

### INTRODUCTION:

➤ This github project offers Python code for determining whether or not a person is likely to have a cardiac condition based on a collection of several criteria specific to each person.

➤ The obtained data was preprocessed using a variety of EDA approaches to find any possible issues, such as missing values, multicollinearity issues, or traits that might be utilised for predicting heart disease. After comparing the effectiveness of several models, including the logistic regression, KNN, random forest, and DNN models, the logistic regression model outperforms all the other models considered.

### MOTIVATION:


<img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/3a77eddc-0f1a-4969-9083-851fefcd8e01">

➤ A major source of death and morbidity, heart disease is a serious worldwide health concern. According to the WHO (Rath et al., 2021) 17.8 million people die from heart disease globally per decade.The evaluation of the patient's medical history, physical examinations, laboratory testing, stress tests, and cardiac catheterisation are frequently combined in the identification and diagnosis of heart disease. These diagnostic techniques, however, could not always give a 
complete picture or might need for intrusive treatments.

➤ Furthermore, heart disease sometimes manifests without symptoms or with mild signs, making it challenging to recognise those who are at risk or who are in the early stages of the disease.

### DATASET:
<img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/9b25fbe6-c50e-4a6e-bbf4-57fbcfb570ed">

The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease.

The dataset has the following variables which we are utilizing for heart disease prediction:

1. **age** - age in years

2. **sex** - (1 = male; 0 = female)

3. **cp** - chest pain type
- **0: Typical angina:** chest pain related decrease blood supply to the heart
- **1: Atypical angina:** chest pain not related to heart
- **2: Non-anginal pain:** typically esophageal spasms (non heart related)
- **3: Asymptomatic:** chest pain not showing signs of disease

**4. trestbps -** resting blood pressure (in mm Hg on admission to the hospital)
**anything above 130-140 is typically cause for concern**

**5. chol -** serum cholestoral in mg/dl
serum = LDL + HDL + .2 * triglycerides
**above 200 is cause for concern**

**6. fbs -** (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
**'>126' mg/dL signals diabetes**

**7. restecg -** resting electrocardiographic results
- **0: Nothing to note**
- **1: ST-T Wave abnormality**
  - can range from mild symptoms to severe problems
  - signals non-normal heart beat
- **2: Possible or definite left ventricular hypertrophy**
  - Enlarged heart's main pumping chamber

**8. thalach -** maximum heart rate achieved

**9. exang -** exercise induced angina (1 = yes; 0 = no)

**10. oldpeak -** ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more

**11. slope -** the slope of the peak exercise ST segment
- **0: Upsloping:** better heart rate with excercise (uncommon)
- **1: Flatsloping:** minimal change (typical healthy heart)
- **2: Downslopins:** signs of unhealthy heart

**12. ca -** number of major vessels (0-3) colored by flourosopy colored vessel means the doctor can see the blood passing through
the more blood movement the better (no clots)

**13. thal -** thalium stress result
- **1,3:** normal
- **6:** fixed defect: used to be defect but ok now
- **7:** reversable defect: no proper blood movement when excercising

**14. target -** have disease or not (1=yes, 0=no) (= the predicted attribute)


### Models Considered:

We have a binary classification Problem since we are trying to predict whether or not a individual has heart disease or not.
Based on this problem Statement we are considering the following models.

- Logistic Regression
- Random Forrest
- k-Nearest Neighbors (k-NN)
- DNN(Direct Neural Network)

### Guidelines to run the code:

1. Download Jupyter Notebook to your local machine.
2. A python version of 3.10.9 is required; if this version is not available, upgrade the python version.
3. Install the necessary dependencies that are listed below using the following commands in jupyter notebook.
- **Regular EDA (exploratory data analysis) and plotting libraries**
    - import numpy as np
    - import pandas as pd
    - import matplotlib.pyplot as plt
    - import seaborn as sns
- **we want our plots to appear inside the notebook**
   - %matplotlib inline 

- **Models from Scikit-Learn**
- Importing Libraries for the models considered: Logistic,K-means clustering,Random Forest Classifier,DNN.
  - from sklearn.linear_model import LogisticRegression
  - from sklearn.neighbors import KNeighborsClassifier
  - from sklearn.ensemble import RandomForestClassifier
  - import tensorflow as tf
  - from sklearn.preprocessing import StandardScaler

- **Importing Libraries for the models Evaluation**
      - from sklearn.model_selection import train_test_split, cross_val_score
      - from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
      - from sklearn.metrics import confusion_matrix, classification_report
      - from sklearn.metrics import precision_score, recall_score, f1_score
      - from sklearn.metrics import roc_curve, auc
4. Run the notebook cells step-by-step to
      - perform EDA
      - Model Implementation
      - Hyper Parameter Tuning for DNN, KNN, Logistic & Random Forrest.
      - Evaluating the best model beyond accuracy namely Sensitivity, Specificity, Recall, and F-1 score by employing cross-validation.
      - Identifying potential features and their contribution to heart disease.
5. To execute each cell, use the run button on top of the Jupyter notebook.

  
