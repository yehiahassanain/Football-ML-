import dython as dython
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from  sklearn import tree
from dython.nominal import associations
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn import svm
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('player-classification.csv')


#Drop the rows that contain missing values
#column
data = data.drop(['national_team','national_rating','national_team_position','national_jersey_number','tags'],axis=1)

#row
data = data.fillna(data.mean())
data = data.fillna(data.mode().iloc[0])

#Features&Label
x=data.iloc[:, 4:86]
label = data['PlayerLevel']

f=x['traits'].str.get_dummies(',')
x = x.join(f)
x=x.drop(['traits'],axis = 1)

e=x['positions'].str.get_dummies(sep=',').rename(lambda x: 'positions_' + x, axis='columns')
x=x.join(e)
x=x.drop(['positions'],axis=1)


cols_split=('LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB')
def split(x, cols):
    for c in cols:
        x[c] = x[c].str.split('+').str[0]
        x[c] = x[c].astype({c: int}, errors='raise')

split(x,cols_split)
x['club_join_date'] = pd.to_datetime(x['club_join_date']).dt.year
x['contract_end_year'] = pd.to_datetime(x['contract_end_year']).dt.year



features=x[['overall_rating', 'potential', 'wage', 'skill_moves(1-5)',
       'release_clause_euro', 'club_rating', 'short_passing', 'dribbling',
       'long_passing', 'ball_control', 'reactions', 'shot_power', 'vision',
       'composure', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM',
       'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM',
       'RDM', 'RWB','club_team']]

features.insert(36,'PlayerLevel',data['PlayerLevel'])

###correlation for cat vars
correlation = associations(
    features, filename='correlation.png',figsize=(10, 10), )
top_feature =correlation['corr'].index[abs(correlation['corr']['PlayerLevel'])>0.52]
top_feature = top_feature.delete(-1)
print(top_feature)

cols=('preferred_foot','body_type','club_position',)
def Feature_Encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x

x=Feature_Encoder(x,cols)

feature_encoder=open("EncoderFile.pickle","wb")
pickle.dump(x,feature_encoder)

#####label
Y = []
for val in label:
    if(val == 'A'):
        Y.append(1)
    elif(val== 'B'):
        Y.append(2)
    elif (val == 'C'):
        Y.append(3)
    elif (val == 'D'):
        Y.append(4)
    else:
        Y.append(5)

Y = np.array(Y)


#encode
encoder1=ce.TargetEncoder()
x['nationality']=encoder1.fit_transform(x['nationality'],x['overall_rating'])




scale_mapper = {"Low/ Low": 0, "Low/ Medium": 1, "Low/ High": 2  ,"Medium/ Low": 3 ,"Medium/ Medium": 4 , "Medium/ High": 5,  "High/ Low": 6,"High/ Medium": 7 ,"High/ High": 8 }
x["work_rate"] = x["work_rate"].replace(scale_mapper)

encoder2=ce.TargetEncoder()
x['club_team']=encoder2.fit_transform(x['club_team'],x['club_rating'])


x=x[top_feature]




#filename='complete_correlation.png'

#logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(x,Y, test_size=0.20,train_size=0.80, random_state=100, shuffle =True)


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
##################################################################################################################################
print("========================================================================================================================")
# Decision tree Model
# spliting data into train set and test set
# training model on training set
classfier_model=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=17,min_samples_leaf=5)

#learnining model on training set
DT_start_Train = time.time()
classfier_model.fit(X_train,y_train)
DT_end_Train = time.time()
#Prediction x_test
Dt_start_Test = time.time()
Training_prediction=classfier_model.predict(X_train)
Test_prediction=classfier_model.predict(X_test)
Dt_end_Test = time.time()
#calculating Accuraccy
DT_Training_Accuracy=accuracy_score(y_train,Training_prediction)
DT_Testing_Accuracy=accuracy_score(y_test,Test_prediction)

Total_train_DT=DT_end_Train-DT_start_Train
Total_test_DT=Dt_end_Test-Dt_start_Test

print("Total training_Time: ",Total_train_DT)
print("Total test_Time: ",Total_test_DT)
print("Accuracy of Training set of Decision_Tree is :",DT_Training_Accuracy*100)
print("Accuracy of Testing tree set of Decision_Tree is :",DT_Testing_Accuracy*100)
print("==========================================================================================================================")

# Decision_tree=open("DecisionTree_file.pickle","wb")
# pickle.dump(classfier_model,Decision_tree)


############################################################################################################################################################################





from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)
KNN_start_Train = time.time()
classifier.fit(X_train, y_train)
KNN_end_Train = time.time()

KNN_start_Test = time.time()
y_pred_train=classifier.predict(X_train)
#KNN_start_Test = time.time()
y_pred_test = classifier.predict(X_test)
KNN_end_Test = time.time()

Total_train_KNN=KNN_end_Train-KNN_start_Train
Total_test_KNN=KNN_end_Test-KNN_start_Test

print("==========================================================================================================================")
print("Total training_Time: ",Total_train_KNN)
print("Total test_Time: ",Total_test_KNN)

from sklearn.metrics import confusion_matrix,accuracy_score
ac_train=accuracy_score(y_train,y_pred_train)
ac_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of train KNN:",ac_train*100)
print("Accuracy of test KNN:",ac_test*100)
print("==========================================================================================================================")
KNN_file=open("KNN_model.pickle","wb")
pickle.dump(classifier,KNN_file)

#print(classifier.classes_)


##########################################################################################################################################################################
#Logistic _linear Model
LogisticRegressionModel = LogisticRegression(penalty='l2',solver='newton-cg',C=50,random_state=33)
Logistic_start_Train = time.time()
LogisticRegressionModel.fit(X_train, y_train)
Logistic_end_Train = time.time()

Logistic_start_Test=time.time()
logistic_Train_predict=LogisticRegressionModel.predict(X_train)
logistic_Test_predict=LogisticRegressionModel.predict(X_test)
Logistic_end_test = time.time()
Train_Accuracy=accuracy_score(y_train,logistic_Train_predict)
Test_accuracy=accuracy_score(y_test,logistic_Test_predict)
Total_training_log=Logistic_end_Train-Logistic_start_Train
Total_test_log=Logistic_end_test-Logistic_start_Test
print("==========================================================================================================================")
print("Total training_Time: ",Total_training_log)
print("Total test_Time: ",Total_test_log)
print("Accuracy of train of logistic:",Train_Accuracy*100)
print("Accuracy of test of logistic:",Test_accuracy*100)
print("==========================================================================================================================")

# Logisticmodel=open("LogisticModel_File.pickle","wb")
# pickle.dump(LogisticRegressionModel,Logisticmodel)
##########################################################################################################################################################################

# # SVM_polynomial-------------------------------------------------------------
# # train the model
# c=0.1
# svm_model=svm.SVC(kernel='',gamma=0.01, C=c).fit(X_train, y_train)
#
# svm_model.fit(X_train,y_train)
# train_predictions=svm_model.predict(X_train)
# test_predictions=svm_model.predict(X_test)
# Svm_Train_Accuracy=accuracy_score(y_train,train_predictions)
# Svm_Test_Accuracy=accuracy_score(y_test,test_predictions)
#
# print("Accuracy of Training set of Svm_poly is :",Svm_Train_Accuracy*100)
# print("Accuracy of Testing tree set of Svm_poly is :",Svm_Test_Accuracy*100)
