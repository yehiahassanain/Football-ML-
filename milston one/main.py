import math
import time
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import bisect
import pickle
from sklearn.compose import make_column_selector as selector
# Load players data
data = pd.read_csv('player-value-prediction.csv')

# Drop the rows that contain missing values
# column
data= data.drop(['national_team', 'national_rating', 'national_team_position', 'national_jersey_number', 'tags'],axis=1)
# Drop_file=open("DropFile.pickle","wb")
# pickle.dump(data,Drop_file)

# Drop_output=open("DropFile.pickle","rb")
# output=pickle.load(Drop_output)
# print(output)
# row
data = data.fillna(data.mean())
data = data.fillna(data.mode().iloc[0])

# Fillna_file=open("Fillna_File.pickle","wb")
# pickle.dump(data,Fillna_file)

# fillna_output=open("Fillna_File.pickle","rb")
# output=pickle.load(fillna_output)
# print(output)



# Features&Label
x = data.iloc[:, 4:86]
Y = data['value']

# encode
encoder1 = ce.TargetEncoder()
x['nationality'] = encoder1.fit_transform(x['nationality'], x['overall_rating'])

cols = ('preferred_foot', 'body_type', 'club_position',)


def Feature_Encoder(x, cols):
    encode={}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
        encode[c]=lbl
    return x,encode


x,enc1 = Feature_Encoder(x, cols)
#print(cols)
#print(enc1.keys())
#print(enc1.values())

#function that handel unknown data
for i in enc1:
    le_classes= enc1[i].classes_.tolist()
    bisect.insort_left(le_classes, 'unknown')
    enc1[i].classes_ = le_classes
    enc1[i].classes_ = np.array(enc1[i].classes_)
print(enc1['body_type'].classes_)

def encoder_transform(X, cols, encoder):
    for c in cols:
        le = encoder[c]
        X[c] = X[c].map(lambda s: 'unknown' if s not in le.classes_ else s)
        X[c] = le.transform(list(X[c].values))

    return X
############################################################################################################################################################

scale_mapper = {"Low/ Low": 0, "Low/ Medium": 1, "Low/ High": 2, "Medium/ Low": 3, "Medium/ Medium": 4,
                "Medium/ High": 5, "High/ Low": 6, "High/ Medium": 7, "High/ High": 8}
x["work_rate"] = x["work_rate"].replace(scale_mapper)

encoder2 = ce.TargetEncoder()
x['club_team'] = encoder2.fit_transform(x['club_team'], x['club_rating'])

f = x['traits'].str.get_dummies(',')
x = x.join(f)
x = x.drop(['traits'], axis=1)

e = x['positions'].str.get_dummies(sep=',').rename(lambda x: 'positions_' + x, axis='columns')
x = x.join(e)
x = x.drop(['positions'], axis=1)

cols_split = (
'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB')


def split(x, cols):
    for c in cols:
        x[c] = x[c].str.split('+').str[0]
        x[c] = x[c].astype({c: int}, errors='raise')


split(x, cols_split)

x['club_join_date'] = pd.to_datetime(x['club_join_date']).dt.year
x['contract_end_year'] = pd.to_datetime(x['contract_end_year']).dt.year

# print(x)

# modeling (polynomial)
# Correlation
correlation = data.corr()
Top_features = correlation.index[abs(correlation['value']) > 0.5]
# Draw correlation between training features(Top 50%)

plt.subplots(figsize=(12, 8))
highest_correlation = data[Top_features].corr()
sns.heatmap(highest_correlation, annot=True)
plt.show()
Top_features = Top_features.delete(-1)
x = x[Top_features]


# scaling
# Normalization

# def Scaling(X, a, b):
#     X = np.array(X)
#     Normalized_X = np.zeros((X.shape[0], X.shape[1]))
#     for i in range(X.shape[1]):
#         Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
#     return Normalized_X
#
#
# x = Scaling(x, 0, 1)

# ===================================================================================================================================

# Test And Train split

X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.20, train_size=0.80, shuffle=False,random_state=10)

from sklearn.preprocessing import MinMaxScaler
def Preprocessing_Scaling(X_train, X_test):
    mx = MinMaxScaler()
    col = []
    for c in X_train.columns:
        if X_train[c].dtype == 'object':
            continue
        else:
            col.append(c)
    X_train_scaled = mx.fit_transform(X_train[col])
    X_test_scaled = mx.transform(X_test[col])
    return mx, X_train_scaled, X_test_scaled

max_scaler,X_train, X_test=Preprocessing_Scaling(X_train, X_test)




poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)

#Saving polynomiol in pickle
poly_model = linear_model.LinearRegression()

# polynomial_model=open("Polynomial_File.pickle","wb")
# pickle.dump(poly_model,polynomial_model)
##########################################################################################################################################################
# traing time
start = time.time()
poly_model.fit(X_train_poly, Y_train)
# Train_Prediction
Y_Train_Prediction = poly_model.predict(X_train_poly)
Y_prediction = poly_model.predict(poly_features.fit_transform(X_test))



# Prediction on test_dataset

Test_Prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Mean Square Error', metrics.mean_squared_error(Y_test, Test_Prediction))
end = time.time()
polynomial_runningtime = end - start
print("training time for polynomial linear regreation is ", polynomial_runningtime)
polynomial_accuracy = r2_score(Y_test, Test_Prediction)
print("accuracy for polynomial linear regression is", polynomial_accuracy * 100)

# modeling (multi_variable)
# =================================================================
# Apply Linear Regression on the selected features
cls = linear_model.LinearRegression()
#Saving Multivariable Linear in pickle

# Multivarible_model=open("Multivariabl_Model_File.pickle","wb")
# pickle.dump(cls,Multivarible_model)


###############################################################################################################################################################
start2 = time.time()
cls.fit(x, Y)
prediction = cls.predict(x)
# Prediction on test_dataset
print("================================================================")
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), prediction))
end2 = time.time()
multivariable_time = end2 - start2
print("training time for multivariable linear regreation is", multivariable_time)
polynomial_accuracy = r2_score(Y, prediction)
print("accuracy for multivariable linear regression is", polynomial_accuracy * 100)