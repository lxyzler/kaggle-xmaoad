#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics

data=pd.read_csv('./train.csv',encoding='utf-8')
test=pd.read_csv('./test.csv',encoding='utf-8')
x=data.drop(columns=data.columns[len(data.columns)-1])
y=data['time'].to_frame(name='')
y.columns = ['time']
test=test.drop(['id'],axis=1)
x=x.drop(['id'],axis=1)
#create new features
def feature(x):
    for col in x.columns:
        if x[col].dtype==object:
            x[col]=LabelEncoder().fit_transform(x[col])
    x=x.drop(['random_state','scale'],axis=1)
    x['n']=x['n_informative']+x['max_iter']+ x['n_samples']+x['n_jobs']+x['alpha']+x['n_features']+x['l1_ratio']+x['flip_y']
    x['n2']=x['n_informative']*x['max_iter']*x['n_samples']*x['n_jobs']*x['alpha']*x['n_features']*x['l1_ratio']*x['flip_y']
    x['n3']=x['n_features']*x['l1_ratio']*x['penalty']
    x['n4']=x['l1_ratio']+x['penalty']
    x['n5']=x['l1_ratio']+x['alpha']+x['flip_y']
    x['n7']=x['l1_ratio']*x['penalty']
    x['n6']=x['n_classes']+x['n_clusters_per_class']
    x['penalty']=x['penalty']*x['penalty']*x['penalty']
    x['l1_ratio']=x['l1_ratio']*10
    x['flip_y']=x['flip_y']*10
    x['alpha']=x['alpha']*10
    x['n']=x['n']*x['n']*x['n']
    x['n2']=x['n2']*x['n2']*x['n2']
    x['n4']=x['n4']+x['n4']+x['n4']
    x['n5']=x['n5']*x['n5']*x['n5']
    x['n6']=x['n6']*x['n6']*x['n6']
    x['n7']=x['n7']*x['n7']*x['n7']
    x['flip_y']=x['flip_y']*x['flip_y']*x['flip_y']
    x['alpha']=x['alpha']*x['alpha']*x['alpha']
    x['l1_ratio']=x['l1_ratio']*x['l1_ratio']*x['l1_ratio']
    x['max_iter']=x['max_iter']*x['max_iter']*x['max_iter']
    x['n_features']=x['n_features']+x['n_features']+x['n_features']
    x['n_samples']=x['n_samples']*x['n_samples']*x['n_samples']
    x['n_classes']=x['n_classes']*x['n_classes']*x['n_classes']
    x['n_clusters_per_class']= x['n_clusters_per_class']*x['n_clusters_per_class']*x['n_clusters_per_class']
    x['n_informative']=x['n_informative']+x['n_informative']+x['n_informative']
    for i in range(len(x['n_jobs'])):
        if x['n_jobs'][i]<0:
            x['n_jobs'][i]=16
    x['n8']=x['n_jobs']+x['penalty']
 
    for col in x.columns:
        x[col]=x[col]+x[col]

     
    return x

x=feature(x)
x.describe()
test=feature(test)



#feature selection
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=300,max_features=2,  random_state=0, n_jobs=-1,max_depth=12) 
forest.fit(x,y) 
feat_labels=x.columns
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(x.shape[1]): 
    #给予10000颗决策树平均不纯度衰减的计算来评估特征重要性 
    #print ("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]) ) 
    #可视化特征重要性-依据平均不纯度衰减
    #plt.bar(range(x.shape[1]),importances[indices],color='lightblue',align='center') 
    #plt.xticks(range(x.shape[1]),feat_labels,rotation=90) 
    #plt.xlim([-1,x.shape[1]]) 
    #plt.tight_layout() 
    #plt.show()
    if importances[indices[f]]<=0.015:
        pass
        del x[feat_labels[f]]
        del test[feat_labels[f]]

#model building
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=50)
forest = RandomForestRegressor(n_estimators=640,max_features=11, random_state=4, n_jobs=-1,max_depth=17) 
forest.fit(x_train,y_train) 
forest_pred = forest.predict(x_test)
kfold = KFold(n_splits=5)
y_test_pred = forest.predict(test)
y_test_pred=pd.DataFrame(y_test_pred)
y_test_pred=y_test_pred.abs()
y_test_pred.index.name='Id'
y_test_pred.columns=['time']
y_test_pred.to_csv('f.csv')
forest.score(x_test,y_test)





