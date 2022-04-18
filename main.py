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

#Load players data
data = pd.read_csv('player-value-prediction.csv')

#Drop the rows that contain missing values
#column
data = data.drop(['national_team','national_rating','national_team_position','national_jersey_number','tags'],axis=1)
#row
data = data.fillna(data.mean())
data=data.fillna(data.mode().iloc[0])

#Features&Label
x=data.iloc[:,4:86]
Y=data['value']


#encode
encoder1=ce.TargetEncoder()
x['nationality']=encoder1.fit_transform(x['nationality'],x['overall_rating'])

cols=('preferred_foot','body_type','club_position',)
def Feature_Encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x

x=Feature_Encoder(x,cols)

scale_mapper = {"Low/ Low": 0, "Low/ Medium": 1, "Low/ High": 2  ,"Medium/ Low": 3 ,"Medium/ Medium": 4 , "Medium/ High": 5,  "High/ Low": 6,"High/ Medium": 7 ,"High/ High": 8 }
x["work_rate"] = x["work_rate"].replace(scale_mapper)

encoder2=ce.TargetEncoder()
x['club_team']=encoder2.fit_transform(x['club_team'],x['club_rating'])

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


#scaling
#Normalization

def Scaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

x = Scaling(x,0,1)

print(x)