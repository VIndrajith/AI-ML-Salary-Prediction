# -*- coding: utf-8 -*-
"""Project_AI/ML.ipynb
**EDA**
"""

# imports necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/adult.csv')

df

df.shape

df.info()

df.describe()

num_col = df.select_dtypes(include=['int64']).columns.tolist()                  # groups all numerical columns
catg_col = df.select_dtypes(include=['object']).columns.tolist()                # groups all categorical columns

print("Numeric Columns:", num_col)
print("Categorical Columns:", catg_col)

# returns all the unique values of each categorical columns
for i in catg_col:
  print("Column: ",i)
  print(df[i].unique())

# bar-plot of 3 categorical columns
plt.figure(figsize=(10, 5))

for i, col in enumerate(['workclass', 'education', 'marital.status']):          # enumerate returns the index pos and the column name
    plt.subplot (1,3, i+1)                                                      # (no. of rows, no. of columns, iteration of each col.)
    df[col].value_counts().plot(kind='bar')                                     # plots bar graph
    plt.title(col)
    plt.xticks(rotation=90)

plt.tight_layout()                                                              # adjust the space btwn the graphs
plt.show()

# returns correlation heatmap
sns.heatmap(df[num_col].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Histograms for 3 numeric columns
df[['age', 'hours.per.week', 'capital.gain']].hist(bins=20, figsize=(10, 8))    # bin -> splits data into 20 bins per col.
plt.tight_layout()                                                              # adjust the space btwn the graphs
plt.show()

"""**Preprocessing**"""

df = df.replace('?',np.nan)                                                     # replace '?' with NaN value
df

#checking for null value count in categorical columns
for i in catg_col:
  print(df[i].isna().value_counts())                                            # false -> no null value
  print()                                                                       # true -> null value

#replaces all null values in categorical col. with mode value
for i in catg_col:
  df[i] = df[i].fillna(df[i].mode()[0])                                         # replace NaN with mode, [0] -> index pos. in case of multiple mode values

df

#returns the number of duplicated rows
df.duplicated().sum()

#drops all duplicated rows
df.drop_duplicates(inplace=True)

#returns the number of duplicated rows
df.duplicated().sum()

plt.figure(figsize=(10, 5))

# Create a boxplot for each numeric column
for i, col in enumerate(num_col, 1):
    plt.subplot(2, 3, i)                                                        # (no. of rows, no. of columns, iteration of each col.)
    sns.boxplot(y=df[col])                                                      # horizontal boxplot
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

for col in num_col:
    Q1, Q3 = np.percentile(df[col], [25, 75])
    IQR = Q3 - Q1
    lb = Q1 - (1.5 * IQR)
    ub = Q3 + (1.5 * IQR)

    # Keep only non-outlier rows
    df = df[(df[col] >= lb) & (df[col] <= ub)]

plt.figure(figsize=(10, 5))

# Create a boxplot for each numeric column
for i, col in enumerate(num_col, 1):
    plt.subplot(2, 3, i)                                                        # (no. of rows, no. of columns, iteration of each col.)
    sns.boxplot(y=df[col])                                                      # horizontal boxplot
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in catg_col:
  df[i] = le.fit_transform(df[i])

df                                                                              # 0 -> Male , 1 -> Female

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['fnlwgt'] = scaler.fit_transform(df[['fnlwgt']])

df

# stores modified data
df.to_csv("m_adult.csv", index=False)

# imports necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/m_adult.csv')               # loading the modified dataset
df

from sklearn.model_selection import train_test_split

X = df.drop('income',axis=1)
y = df['income']

# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 20% Testing 75 and Training (80%)

print(X.shape)
print(y.shape)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

import joblib                                                                   # library used to save the trained model

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
LR_pred = model.predict(X_test)

joblib.dump(model,'LR.pkl')

"""KNN"""

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
KNN_pred = model.predict(X_test)

joblib.dump(model,'KNN.pkl')

"""SVM"""

from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
SVM_pred = model.predict(X_test)

joblib.dump(model,'SVM.pkl')

"""Naive Bayes"""

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
NB_pred = model.predict(X_test)

joblib.dump(model,'NB.pkl')

"""Decesion Tree"""

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
DT_pred = model.predict(X_test)

joblib.dump(model,'DT.pkl')

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
RF_pred = model.predict(X_test)

joblib.dump(model,'RF.pkl')

"""MLP"""

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(10,10,5), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
MLP_pred = model.predict(X_test)

joblib.dump(model,'MLP.pkl')

"""Gradient Boosting"""

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
GB_pred = model.predict(X_test)

joblib.dump(model,'GB.pkl')

"""XGB"""

from xgboost import XGBClassifier

model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)
XGB_pred = model.predict(X_test)

joblib.dump(model,'XGB.pkl')

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, LR_pred))
print(classification_report(y_test, LR_pred))

cm = confusion_matrix(y_test, LR_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("k-Nearest Neighbors")
print("Accuracy:", accuracy_score(y_test, KNN_pred))
print(classification_report(y_test, KNN_pred))

cm = confusion_matrix(y_test, KNN_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - kNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Support Vector Machine")
print("Accuracy:", accuracy_score(y_test, SVM_pred))
print(classification_report(y_test, SVM_pred))

cm = confusion_matrix(y_test, SVM_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" Naive Bayes")
print("Accuracy:", accuracy_score(y_test, NB_pred))
print(classification_report(y_test, NB_pred))

cm = confusion_matrix(y_test, NB_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" Decision Tree")
print("Accuracy:", accuracy_score(y_test, DT_pred))
print(classification_report(y_test, DT_pred))

cm = confusion_matrix(y_test, DT_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" Random Forest")
print("Accuracy:", accuracy_score(y_test, RF_pred))
print(classification_report(y_test, RF_pred))

cm = confusion_matrix(y_test, RF_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" MLP Classifier")
print("Accuracy:", accuracy_score(y_test, MLP_pred))
print(classification_report(y_test, MLP_pred))

cm = confusion_matrix(y_test, MLP_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - MLP Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" Gradient Boosting")
print("Accuracy:", accuracy_score(y_test, GB_pred))
print(classification_report(y_test, GB_pred))

cm = confusion_matrix(y_test, GB_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(" XGBoost Classifier")
print("Accuracy:", accuracy_score(y_test, XGB_pred))
print(classification_report(y_test, XGB_pred))

cm = confusion_matrix(y_test, XGB_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""Hyperparamaeter tuning (Mannual Tuning)"""

# kNN hyperparameter tuning
for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    KNN_pred = model.predict(X_test)

    print(f"k-Nearest Neighbors for {k}")
    print("Accuracy:", accuracy_score(y_test, KNN_pred))
    print(classification_report(y_test, KNN_pred))

# SVM hyperparameter tuning
for k in ['linear', 'rbf', 'poly']:
    model = SVC(kernel=k)
    model.fit(X_train, y_train)
    SVC_pred = model.predict(X_test)

    print(f"Support Vector Machine for k={k}")
    print("Accuracy:", accuracy_score(y_test, SVM_pred))
    print(classification_report(y_test, SVM_pred))

# Random-Forest hyperparameter tuning
for n in [50, 100, 150]:
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    RF_pred = model.predict(X_test)

    print(f"Random Forest for n={n}")
    print("Accuracy:", accuracy_score(y_test, RF_pred))
    print(classification_report(y_test, RF_pred))

"""### 🔍 Classifier Performance Comparison

| **Model**             | **Hyperparameter(s)** | **Accuracy (%)** | **Precision** | **Recall** | **F1-Score** |
|-----------------------|------------------------|------------------|---------------|------------|--------------|
| Logistic Regression   | —                      | 81.3             | 0.79          | 0.81       | 0.78         |
| kNN                   | —                      | 80.3             | 0.79          | 0.80       | 0.80         |
| SVM                   | —                      | 78.6             | 0.62          | 0.79       | 0.69         |
| Naive Bayes           | —                      | 78.3             | 0.81          | 0.78       | 0.79         |
| Decision Tree         | —                      | 78.0             | 0.78          | 0.78       | 0.78         |
| Random Forest         | —                      | 82.97            | 0.82          | 0.83       | 0.82         |
| MLP Classifier        | —                      | 83.36            | 0.82          | 0.83       | 0.82         |
| Gradient Boosting     | —                      | 84.50            | 0.83          | 0.84       | 0.83         |
| XGBoost Classifier    | —                      | 83.18            | 0.82          | 0.83       | 0.82         |

<br>

#### ✅ Summary:
- **Gradient Boosting** achieved the highest **accuracy (84.5%)** and strong overall performance.
- **MLP, XGBoost, and Random Forest** also performed competitively, all with F1-scores around **0.82–0.83**.
- **Naive Bayes and Decision Tree** showed balanced results but slightly lower than ensemble methods.
- **SVM** underperformed in this case, with significantly lower precision and F1-score due to failing to classify class `1`.

### 🔧 Classifier Performance with Manual Hyperparameter Tuning

| **Model**       | **Hyperparameter(s)**   | **Accuracy (%)** | **Precision** | **Recall** | **F1-Score** |
|------------------|--------------------------|------------------|---------------|------------|--------------|
| kNN              | k = 3                    | 79.7             | 0.79          | 0.80       | 0.79         |
| kNN              | k = 5                    | 80.3             | 0.79          | 0.80       | 0.80         |
| kNN              | k = 7                    | 80.6             | 0.79          | 0.81       | 0.80         |
| kNN              | k = 9                    | 80.8             | 0.79          | 0.81       | 0.80         |
| SVM              | kernel = 'linear'        | 78.6             | 0.62          | 0.79       | 0.69         |
| SVM              | kernel = 'rbf'           | 78.6             | 0.62          | 0.79       | 0.69         |
| SVM              | kernel = 'poly'          | 78.6             | 0.62          | 0.79       | 0.69         |
| Random Forest    | n_estimators = 50        | 82.8             | 0.82          | 0.83       | 0.82         |
| Random Forest    | n_estimators = 100       | 82.7             | 0.82          | 0.83       | 0.82         |
| Random Forest    | n_estimators = 150       | 82.5             | 0.81          | 0.82       | 0.82         |

<br>

##📌 Observations:

- kNN improves slightly from k=3 to k=9, peaking at ~80.8% accuracy.

- SVM performance is poor across all kernels, due to zero predictions for class 1.

- Random Forest performs consistently well, with the best accuracy (~82.8%) at n=50.
"""