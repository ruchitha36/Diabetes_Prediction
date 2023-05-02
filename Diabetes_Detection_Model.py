import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv(r'\diabetes_data.csv')
data.head(3)

## Rename column names 
data.columns =['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age','Class']

data.info()
data.describe()

data.Class.value_counts(normalize=True)*100

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN baacuse it cannot be 0.
df_model = data.copy(deep=True)
df_model[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_model[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## Total null values in each column of dataframe
df_model.isnull().sum()


# Replacing NaN value by mean, median depending upon distribution
df_model['Glucose'].fillna(df_model['Glucose'].mean(), inplace=True)
df_model['BloodPressure'].fillna(df_model['BloodPressure'].mean(), inplace=True)
df_model['SkinThickness'].fillna(df_model['SkinThickness'].median(), inplace=True)
df_model['Insulin'].fillna(df_model['Insulin'].median(), inplace=True)
df_model['BMI'].fillna(df_model['BMI'].median(), inplace=True)

# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df_model['Target']= label_encoder.fit_transform(df_model['Class']) 

## Drop class column 
df_model.drop('Class', inplace=True, axis=1)

#skewness of columns
for i in df_model.columns:
    print("The skewness of "+ i +" is {}".format(df_model[i].skew()))

## Plot histogram of numeric features  
df_model.hist(bins=20, figsize=(15,8))

## Create a heatmap of correlation matrix between features.
df_model.corr()
sns.heatmap(df_model.corr(), annot=True)

## Model Development

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

X = df_model.drop(columns=['Target'], axis=1)
y = df_model['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))

# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Creating a confusion matrix for test set
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_pred = classifier.predict(X_test)
con_mat = confusion_matrix(y_test, y_pred)
con_mat

# Plotting the confusion matrix
plt.figure(figsize=(10,7))
p = sns.heatmap(con_mat, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion matrix for Random Forest Classifier Model Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# Accuracy Score
score = round(accuracy_score(y_test, y_pred),4)*100
print("Accuracy on test set: {}%".format(score))

# Classification Report
print(classification_report(y_test, y_pred))

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


