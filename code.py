# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:24:26 2024

@author: BTG
"""

# =============================================================================
# Author : 배태겸
# 
# Subject : 품질 이상탐지/진단(도금욕) AI 데이터셋
# =============================================================================



# 1. 데이터 준비
# 1-1. 라이브러리/패키지 설치
# pip install pandas
# pip install numpy
# pip install sklearn
# pip install autokeras
# pip install seaborn
# pip install tensorflow
# pip install datetime
# pip install matplotlib
# pip install pydot
# pip install graphviz


# 1-2. 라이브러리/패키지 불러오기
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import tree
import autokeras as ak
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import tensorflow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report






# 1-3. 데이터 불러오기
root_dir = os.getcwd()
f_lists = os.listdir(root_dir)
print("File Lists : ", f_lists)


new_file_lists = [f for f in f_lists if f.endswith('.csv')]
print("File Lists : ", new_file_lists)



data_lists = new_file_lists[:-1]
error_list = new_file_lists[0]
print("Data Lists : ", data_lists)
print("Error Data List : ", error_list)


def csv_read_(data_dir, data_list):
    tmp = pd.read_csv(os.path.join(data_dir, data_list), sep=',', encoding='utf-8')
    y, m, d = map(int, data_list.split('-')[-1].split('.')[:-1])
    time = tmp['Time']
    tmp['DTime'] ='-'.join(data_list.split('-')[-1].split('.')[:-1])
    ctime = time.apply(lambda _ : _.replace(u'오후', 'PM').replace(u'오전', 'AM'))
    n_time = ctime.apply(lambda _ : datetime.datetime.strptime(_, "%p %I:%M:%S.%f"))
    newtime = n_time.apply(lambda _ : _.replace(year=y, month=m, day=d))
    tmp['Time'] = newtime
    
    return tmp


dd = csv_read_(root_dir, data_lists[0])

for i in range(1, len(data_lists)):
    dd = pd.merge(dd, csv_read_(root_dir, data_lists[i]), how='outer')
dd


dd = dd.drop('Index', axis=1)
dd


dd = dd.set_index('Time')
dd





# 2. 데이터 탐색
# 2-1. 데이터 사본 생성
dedicated_data = dd.copy()
dedicated_data

# 2-2. 데이터 컬럼 확인
dedicated_data.columns

# 2-3. 데이터 프레임 요약
dedicated_data.describe()

# 2-4. 데이터프레임 크기 확인
dedicated_data.shape

# 2-5. 데이터프레임 정보 확인
dedicated_data.info()

# 2-6. 데이터프레임 null값 개수 카운트
dedicated_data.isna().sum()

# 2-7. 데이터 시각화를 통한 histogram 확인
dedicated_data.hist(figsize=(10,10))

# 2-8. 데이터 상관관계 분석
correlation = dedicated_data.corr()
correlation

# 2-9. 데이터 상관관계 시각화
sns.heatmap(correlation, annot=True, fmt='.2f')




# 3. 데이터 정제(전처리)
# 3-1. null값 제거
dedicated_data = dedicated_data.dropna()
dedicated_data


# 4. 알고리즘 선택



# 5. 학습, 평가 데이터 준비
# 5-1. Lot List 추출
lot_lists = dedicated_data['Lot'].unique()
print(lot_lists)
print(len(lot_lists))

# 5-2. Date List 추출
d_lists = dedicated_data['DTime'].unique()
print(d_lists)
print(len(d_lists))


# 5-3. Error Data Read
error = pd.read_csv(os.path.join(root_dir, error_list), sep=',', encoding='utf-8')
error

error_drop = error.dropna()
error_drop


# 5-4. Error Data List 추출
lot_error_lists = error_drop['LoT'].unique()
d_error_lists = error_drop['Date'].unique()
print("Unique LoT List : ", lot_error_lists)
print("Unique Date List : ", d_error_lists)


# 5-5. Train/Test Data Set Make
X_data = pd.DataFrame(columns={'pH','Temp','Current', 'Voltage', 'QC'})



for d in d_lists:
    for lot in lot_lists:
        tmp = dd[(dd['DTime']==d)&(dd['Lot']==lot)]
        tmp = tmp[['pH', 'Temp','Current', 'Voltage']]
        error_df = error_drop[(error_drop['Date']==d)&((error_drop['LoT']==lot))]
        len_error = len(error_df)
        
        if len_error>0:
            trr = np.full((tmp['pH'].shape), 0)
        else:
        trr = np.full((tmp['pH'].shape), 1)
    tmp['QC'] = trr
    X_data = X_data.append(tmp)
X_data=X_data.apply(pd.to_numeric)



train_data, test_data = train_test_split(X_data, test_size=0.2)

train_data.describe()

train_data.corr()

fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(train_data.corr(),
 cmap='RdYlBu_r',
 annot = True,
 linewidths=0.5,
 cbar_kws={"shrink":.5},
 vmin=-1, vmax=1)
plt.show()


test_data.describe()

test_data.corr()

fig, ax = plt.subplots(figsize=(7,7))

sns.heatmap(test_data.corr(),
 cmap='RdYlBu_r',
 annot = True,
 linewidths=0.5,
 cbar_kws={"shrink":.5},
 vmin=-1, vmax=1)
plt.show()


# 6. 모델링
# 6-1. Regression 모델 모델링
is_training = True
if is_training:
    reg = ak.StructuredDataRegressor( overwrite=True, max_trials=5)
    reg.fit(train_data[['pH','Temp','Current', 'Voltage']], train_data[['QC']], verbose=2, epochs=7)
    model = reg.export_model()
else:
    model = tensorflow.keras.models.load_model("structured_data_regressor/best_model", custom_objects=ak.CUSTOM_OBJECTS)


# 6-2. DecisionTree 모델 모델링
clf = tree.DecisionTreeRegressor()


# 7. 모델훈련
# 7-1. Regression 모델 학습
reg.fit(X_data[['pH','Temp','Current']], X_data[['QC']], verbose=2, epochs=10)


# 7-2. Regression 모델 시각화
model.summary()

plot_model(model)


# 7-3. Decision Tree 모델 학습
clf = clf.fit(X_data[['pH','Temp','Current', ‘Voltage’]], X_data[['QC']])


# 7-4. Decision Tree 모델 시각화
vis = True
if vis:
    plt.figure(figsize=(10,30))
    tree.plot_tree(clf)
    plt.show()


# 8. 모델 튜닝
# 8-1. Regression 모델 튜닝


# 8-2. Decision Tree 모델 튜닝
new_clf = tree.DecisionTreeRegressor(max_depth=3)
new_clf = new_clf.fit(train_data[['pH','Temp','Current', ‘Voltage’]], train_data[['QC']])

plt.figure(figsize=(10,10))
tree.plot_tree(new_clf)
plt.show()


# 9. 모델 평가 및 해석
# 9-1. Regression 모델 평가
ak_model_predicted = model.predict(test_data[['pH','Temp','Current', 'Voltage']])
print('AutoKeras Model Predict : ', ak_model_predicted)
rmse = sqrt(mean_squared_error(test_data['QC'], ak_model_predicted))
print('AutoKeras Model RMSE : ',rmse)


y_test = test_data['QC']
y_pred = [round(y[0], 0) for y in ak_model_predicted]
print("accuracy = ", accuracy_score(y_test, y_pred))
print("recall = ", recall_score(y_test, y_pred))
print("precision = ", precision_score(y_test, y_pred))
print("f1 score = ", f1_score(y_test, y_pred))


def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print("TP : ", TP)
print("FP : ", FP)
print("FN : ", FN)
print("TN : ", TN)

if (TP+FN) == 0:
    tpr_val = 0
else:
    tpr_val = TP / (TP+FN)
if (TN+FP) == 0:
    fpr_val = 0
else:
    fpr_val = TN / (TN+FP)
print(tpr_val, fpr_val)
tpr, fpr, _ = roc_curve(y_test, y_pred)
tpr[1] = tpr_val
fpr[1] = fpr_val
if len(tpr) < 3:
 tpr = np.append(tpr, 1)
 fpr = np.append(fpr, 1)
print(fpr, tpr)


plt.plot(tpr, fpr, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.plot([tpr_val], [fpr_val], 'ro', ms=10)
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.grid()
plt.legend()
plt.show()


print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))


fig = plt.figure(figsize=(30, 30))
ax1 = fig.add_subplot(3,2,1, projection='3d')
ax1.scatter(test_data['pH'], test_data['Temp'], ak_model_predicted)
ax1.set_xlabel('pH Test Data')
ax1.set_ylabel('Temp Test Data')
ax1.set_zlabel('Estimated Process Data')
ax2 = fig.add_subplot(3,2,2, projection='3d')
ax2.scatter(test_data['Temp'], test_data['Current'], ak_model_predicted)
ax2.set_xlabel('Temp Test Data')
ax2.set_ylabel('Current Test Data')
ax2.set_zlabel('Estimated Process Data')
ax3 = fig.add_subplot(3,2,3, projection='3d')
ax3.scatter(test_data['Current'], test_data['Voltage'], ak_model_predicted)
ax3.set_xlabel('pH Test Data')
ax3.set_ylabel('Voltage Test Data')
ax3.set_zlabel('Estimated Process Data')
ax4 = fig.add_subplot(3,2,4, projection='3d')
ax4.scatter(test_data['Voltage'], test_data['pH'], ak_model_predicted)
ax4.set_xlabel('Voltage Test Data')
ax4.set_ylabel('pH Test Data')
ax4.set_zlabel('Estimated Process Data')
ax5 = fig.add_subplot(3,2,5, projection='3d')
ax5.scatter(test_data['pH'], test_data['Current'], ak_model_predicted)
ax5.set_xlabel('pH Test Data')
ax5.set_ylabel('Current Test Data')
ax5.set_zlabel('Estimated Process Data')
ax6 = fig.add_subplot(3,2,6, projection='3d')
ax6.scatter(test_data['Temp'], test_data['Voltage'], ak_model_predicted)
ax6.set_xlabel('Temp Test Data')
ax6.set_ylabel('Voltage Test Data')
ax6.set_zlabel('Estimated Process Data')
plt.tight_layout()
plt.show()
