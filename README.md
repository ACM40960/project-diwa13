<p align="center"><h1>🫀PREDICTION OF HEART DISEASE USING MACHINE LEARNING:</h1></p>

### ⚡️INTRODUCTION:

 ➤ This github project offers Python code for determining whether or not a person is likely to have a cardiac condition based on a collection of several criteria specific to each person.

 ➤ The obtained data was preprocessed using a variety of EDA approaches to find any possible issues, such as missing values, multicollinearity issues, or traits that might be utilised for predicting heart disease. After comparing the effectiveness of several models, including the logistic regression, KNN, random forest, and DNN models, the logistic regression model outperforms all the other models considered.


### 🔎METHEDOLOGY:

<p align="center"><img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/ecd197c6-71ff-4b5f-87c0-f9fb5624d817"></p>
<p align="center">FIG : Work Flow </p>
➤ We looked at four models to predict cardiac disease using machine learning. To determine whether or not a person has cardiac disease, algorithms like K means, Random Forrest,DNN and logistic regression have been used. All the four models are most effective at forecasting binary outcomes.

➤ Before modelling the data, we performed exploratory data analysis to identify any discrepancies in the data and apply the appropriate pretreatment measures.

➤ An essential factor that has to be considered is tuning the hyperparameters in all three models. The logistic regression probability value, the number of decision trees,max_depth,min_samples_split,min_samples_leaf in the random forest, In DNN batch_size','epochs','learning_rate','units'  and the number of clusters in the k-means algorithm are the hyperparameters that are tuned using cross-validation methods.

➤ The best model is chosen and applied to predict the presence or absence of heart disease, and its performance is evaluated beyond accuracy.


### 🧠 MOTIVATION:


<p align="center"><img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/df86c154-95b8-4df7-b80f-eaf56fee2980"></p>


➤ A major source of death and morbidity, heart disease is a serious worldwide health concern. According to the WHO (Rath et al., 2021) 17.8 million people die from heart disease globally per decade.The evaluation of the patient's medical history, physical examinations, laboratory testing, stress tests, and cardiac catheterisation are frequently combined in the identification and diagnosis of heart disease. These diagnostic techniques, however, could not always give a 
complete picture or might need for intrusive treatments.

➤ Furthermore, heart disease sometimes manifests without symptoms or with mild signs, making it challenging to recognise those who are at risk or who are in the early stages of the disease.

### 🚀 WHO MAY UTILISE THIS REPO:

- **Data scientists and enthusiasts of machine learning:** The repository may be used as a learning tool for data scientists and machine learning professionals. They can examine the source code, comprehend the machine learning process, and use related methods in their own projects.
- **Healthcare Professionals:** Doctors and medical researchers, among others, can gain from learning how predictive models might help with heart disease diagnosis and prediction. They could be curious about the effectiveness of the approach and how it might affect patient care.
- **Students and Researchers:** Students working on projects and doing research in the disciplines of data science, machine learning, and medicine can utilise the repository as a useful reference. It may be used as a reference to comprehend how machine learning is used in healthcare.


### 📜GOAL OF THE PROJECT:

- The ultimate objective of heart disease prediction using machine learning is to maximise the potential of data analysis and predictive modelling to identify individuals at risk of heart diseases based on their medical records, increase the precision of risk assessment, and advance medical research and healthcare practises in the field of cardiovascular diseases.

### 🛠️APPLICATIONS OF THE PROJECT:

- **Healthcare Organisations:** The repository can be used as a starting point by healthcare organisations wanting to deploy predictive models for the detection and prevention of heart disease. They could modify the model to fit their specific needs and datasets.
- **Health Insurance:** Predictive models may be used by insurance firms to analyse risk profiles and customise insurance policies.
- **Telemedicine:** Predictive models may be integrated into telemedicine systems to offer virtual consultations with risk evaluations.
- **Public Health Surveillance:** By offering information on disease prevalence and trends, aggregated data from prediction models may support public health surveillance.
- **Medical Research:** Large medical datasets may be analysed using machine learning to find new risk factors and relationships, advancing medical research and the creation of novel medicines.
  
### 🎯 DATASET:
<p align="center"><img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/459044ab-a4f4-476b-9eb5-79a03a732492"></p>
<p align="center">TABLE : Snapshot of the dataset</p>
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


### 🤖 Models Considered:

We have a binary classification Problem since we are trying to predict whether or not a individual has heart disease or not.
Based on this problem Statement we are considering the following models.

- Logistic Regression
       - Default values in scikit learn has been used as parameters for logistic Regression
- Random Forrest
       - Default values in scikit learn has been used as parameters for Random Forrest.
- k-Nearest Neighbors (k-NN)
       - Default values in scikit learn has been used as parameters for K-Nearest Neighbour
- DNN(Direct Neural Network):
- A DNN has been constructed with Single hidden layer comprising 16 units has been fitted.
- Relu has been used as an activation function because ReLU helps mitigate the vanishing gradient problem, which is a challenge in deep networks
- The output function consists of a single unit and sigmoid is used as an activation function since its a binary classification.

### ⚙️ Guidelines to run the code:

1. Download Jupyter Notebook to your local machine. You can download Anaconda[Anaconda](https://www.anaconda.com/download) and install Jupyter Noebook. 
2. A python version of 3.10.9 is required; if this version is not available, upgrade the python version. Python can be downloaded from the Link.[Link for downloading Python](https://www.python.org/downloads/)
3. Download the dataset available in our repository and store it in your local directory where jupyter notebook is running.
4. Before running the jupyter notebook make sure to install the following libraries since they have been used for analysis,model implementation,hypeparameter tuning and Evaluation.

**Run the following commands for installing the aforementioned libraries**

```code
pip install numpy
pip install pandas    
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tensorflow
```

4. Run the notebook cells step-by-step to
      - perform Exploratory Data Analysis.
      - Model Implementation
      - Hyper Parameter Tuning for DNN, KNN, Logistic & Random Forrest.
      - Evaluating the best model beyond accuracy namely Sensitivity, Specificity, Recall, and F-1 score by employing cross-validation.
      - Identifying potential features and their contribution to heart disease.
5. To execute each cell, use the run button on top of the Jupyter notebook.

### 💡Different Components of the code:

**1.EDA:**


<p align="center"><img width="586" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/9b246758-cb03-4f25-a59d-2fba6bae0389"></p>
<p align="center">Fig: Culmination of different Exploratory data Analysis performed</p>

- The first segment of the code is to import the Dataset and analyse the Data. Data Analysing is an important process before fitting the data to the model as it helps us understand our data, its structure, and its content.
- EDA helps us to identify missing values, outliers, and anomalies in your dataset. Cleaning and preparing the data is crucial for accurate analysis.
- EDA can assist in identifying which features (variables) are relevant for our analysis.
- EDA aids in our comprehension of the nature of the data and the connections between variables. This knowledge helps us choose the right models for our analysis.

**The above mentioned process are achieved through a serious of steps**
- For finding suitable evaluation metrics for evaluating a model the distribution of the output is analysed via barplot.
- Python predefined methods provided for pandas dataframe are used to find missing values.
- The distribution of data is understood using python predefined methods.
- The potential features contributing to prediction of heart disease is achieved through a series of bar and scatter plots.
- The multicollinearity problem is identified using correlation plot obtained using predefined method from pandas library.

**2. Model Implementation:**
- The second section of the code is implementing the selected models and comparing the models based on accuracy.
   The aforementioned is achieved through a series of processes.
- The train, test split is achieved through sklearn.model_selection import train_test_split.
- The models namely logistic, KNN, and Randomforrest for fitting the data are achieved using python predefined library sklearn.
- The DNN model is created with the help of tensorflow.
  
**3. Hyper Parameter Tuning:**
- The third section of the code is about hyperparameter tuning
- For DNN, The following packages are imported to perform hyper tuning. from tensorflow.keras.models import Sequential,from tensorflow.keras.layers import Dense, from TensorFlow.keras.wrappers.scikit_learn import KerasClassifier, from tensorflow.keras.optimizers import Adam.
- For logistic and Random forest the hyperparameters are tuned using random search. This is made possible using the following predefined library from sklearn.model_selection import RandomizedSearchCV.
- Each optimised model's accuracy is computed and compared.
  
**4. Evaluting our tuned machine learning classifier for the best model obtained beyond accuracy**
- The fourth section of code is about evaluating the best model obtained to evaluate beyond accuracy by cross validation.
- The confusion matrix is obtained using predefined function from sklearn.metrics library import confusion_matrix 
- The precision,recall,f1 score is calculated using predefined functions imported from sklearn.metrics library import precision_score, recall_score, f1_score
- The roc curve and auc score are obtained from predefined sklearn.metrics library using roc_curve function.

### RESULTS:

<p align="center"><img width="400" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/fed8f367-6158-4402-ac5a-55dc2cf304ce">
</p>
<p align="center">Fig: Comparison of models based on Accuracy</p>

- We have considered 4 models namely Logistic, Random forest, KNN,DNN.Based on the evaluation of four models (Logistic Regression, Random Forest, K-Nearest Neighbour's, Deep Neural Network), after hyperparameter tuning and fitting the optimised models on balanced data, Logistic Regression emerged as the best-performing model based on accuracy. 

<p align="center"><img width="360" alt="dataset pic 1" src="https://github.com/ACM40960/project-diwa13/assets/115154682/90a65538-8303-4d94-84ce-76c49fb42c59"></p>
<p align="center">Fig: Evaluation of Logistic Regression based on Precison,F1 score,Recall,Confusion matrix,Roc Curve</p>


- Subsequently, the evaluation of the Logistic Regression model included the following metrics to gauge its predictive capacity:
    - Area Under the ROC Curve (AUC):(0.93)
    - Recall(True Positive Rate):(0.92)
    - Precision:(0.82)
    - F1 Score:(0.86)

- These metrics collectively offer a comprehensive understanding of the model's performance across various aspects. While accuracy is a valuable metric, these additional measures give deeper insights into how well the model performs under different scenarios and classes. Based on the evaluated metrics, the Logistic Regression model's AUC, Sensitivity, Specificity, and F1 Score provide a more nuanced assessment of its predictive capacity. We may thus draw the conclusion that, when compared to other models, logistic regression has greater predictive ability.
  
### 💡CONTRIBUTION
- We encourage contributions to this repository. Feel free to send in a pull request if you have ideas for improved machine learning models, want to fine-tune the hyperparameters, identify any data abnormalities, or want to improve any aspect of the project.

## 🛠️CREDITS:
- This project is in collaboration with SakethSaiNigam (https://github.com/ACM40960/project-SakethSaiNigam)
- If you have any queries you can mail them to us:
   - Mail id: Diwakar Mohan (mohann111122@gmail.com)
   - Mail id : SakethSaiNigam (saketh1506@gmail.com)

## 🔗 REFERENCES:

- Machine Learning Technology-Based Heart Disease Detection Models, Authors:Umarani Nagavelli, Debabrata Samanta, and Partha Chakraborty[Link1](https://www.hindawi.com/journals/jhe/2022/7351061/)
- Effective Heart Disease Prediction Using Machine Learning Techniques, Authors:Chintan M. Bhatt ,Parth Patel,Tarang Ghetia and Pier Luigi Mazzeo[Link2](https://www.mdpi.com/1999-4893/16/2/88)
- Heart disease prediction using machine learning algorithms, Authors: Harshit Jindal, Sarthak Agrawal, Rishabh Khera, Rachna Jain and Preeti Nagrath Published under licence by IOP Publishing Ltd [Link3](https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012072)
- Diagnosis And Prediction Of Heart Disease Using Machine Learning Techniques, Author:J.Jeyaganesan, A.Sathiya , S.Keerthana, Aaradhyanidhi Aiyer[Link4](https://www.bibliomed.org/?mno=141030)
-  A. L. Bui, T. B. Horwich, and G. C. Fonarow, ‘‘Epidemiology and risk profile of heart failure,’’ Nature Rev. Cardiol., vol. 8, no. 1, p. 30, 2011.[Link5](https://pubmed.ncbi.nlm.nih.gov/21060326/)




