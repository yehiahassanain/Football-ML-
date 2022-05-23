import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#import category_encoders as ce
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle

data = pd.read_csv('player-test-samples.csv')

#drop_column
data = data.drop(['national_team','national_rating','national_team_position','national_jersey_number','tags','club_team','traits','nationality'],axis=1)


#miss_in_row
data = data.fillna(data.mean())
data=data.fillna(data.mode().iloc[0])


#Features&Label
x=data.iloc[:, 4:83]
label =data['PlayerLevel']

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


#label_encoder
file = open("dict_all.obj", 'rb')
dict_all_loaded = pickle.load(file)
file.close()
cols=('preferred_foot','body_type','club_position','work_rate',)
for col in cols:
  x[col].replace(dict_all_loaded[col], inplace=True)

print(dict_all_loaded)



#=========================
#The next function will add missing columns (in response to df_train)
def add_missing_dummy_columns(df, columns):
  missing_cols = set(columns) - set(df.columns)
  for c in missing_cols:
    df[c] = 0

#The next function will delete extra columns (in response to df_train)
def fix_columns(df, columns):
  add_missing_dummy_columns(df, columns)
# make sure we have all the columns we need
  assert(set(columns) - set(df.columns) == set())
  extra_cols = set(df.columns) - set(columns)
  if extra_cols:
    df = df[columns]
  return df

file = open("positions_cols.obj", 'rb')
positions_cols_loaded = pickle.load(file)
file.close()

#Execute get_dummies to One-Hot
e=x['positions'].str.get_dummies(sep=',').rename(lambda x: 'positions_' + x, axis='columns')
#Run the above functions to pad the deploy dataset
fixed_d = fix_columns(e, positions_cols_loaded)
x=x.join(fixed_d)
x=x.drop(['positions'],axis=1)

#=================

#remove(+2)from column
cols_split=('LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB')
def split(x, cols):
    for c in cols:
        x[c] = x[c].str.split('+').str[0]
        x[c] = x[c].astype({c: int}, errors='raise')

split(x,cols_split)

#date
x['club_join_date'] = pd.to_datetime(x['club_join_date']).dt.year
x['contract_end_year'] = pd.to_datetime(x['contract_end_year']).dt.year




###still have string colum drop

# load the scaler
#scaler =pickle.load(open('C:/Users/Fatima/Desktop/project machin/Phase2_project/Preprocessing_Scaling.pkl', 'rb'))
#X_test_scaled = scaler.transform(x)

# scaler = MinMaxScaler()
# scaler.fit(x)
# print(x)

# load the scaler
file_saver=open("C:/Users/yehia/Documents/GitHub/Football-ML-/milston two/Phase2_project/save_dic.pickle","rb")
saver = pickle.load(file_saver)
x=x[saver]
scaler =pickle.load(open('C:/Users/yehia/Documents/GitHub/Football-ML-/milston two/Phase2_project/Preprocessing_Scaling.pkl', 'rb'))
X_test_scaled = scaler.transform(x)

model1 = open("C:/Users/yehia/Documents/GitHub/Football-ML-/milston two/Phase2_project/DecisionTree_file.pickle","rb")

model = pickle.load(model1)
test_pred=model.predict(X_test_scaled)
DT_Testing_Accuracy=accuracy_score(Y,test_pred)
print("Accuracy of Testing tree set of Decision_Tree is :",DT_Testing_Accuracy*100)
