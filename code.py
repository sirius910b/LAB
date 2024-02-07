# =============================================================================
# Author : 배태겸
# Date : 2024.02.07
# Subject : 품질 이상탐지/진단(도금욕) AI 데이터셋
# =============================================================================

pip uninstall numpy
pip install numpy

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
root_dir = os.getcwd() # 현재 작업 디렉토리
f_lists = os.listdir(root_dir) # 해당 디렉토리내 모든 파일 리스트 가져오기
print("File Lists : ", f_lists)


new_file_lists = [f for f in f_lists if f.endswith('.csv')]
print("File Lists : ", new_file_lists)


data_lists = new_file_lists[:-1]
error_list = new_file_lists[0]
print("Data Lists : ", data_lists)
print("Error Data List : ", error_list)


def csv_read_(data_dir, data_list):
    tmp = pd.read_csv(os.path.join(data_dir, data_list), sep=',', encoding='utf-8') 
    y, m, d = map(int, data_list.split('-')[-1].split('.')[:-1]) # 파일 이름에서 날짜 정보를 추출. 파일 이름이 'yyyy-mm-dd.csv' 형태라고 가정할 때, 이 구문은 파일 이름을 먼저 '-'로 분리한 후, 마지막 요소([-1])인 'dd.csv'를 다시 '.'으로 분리하여 'dd' 부분만 추출한 다음, 연(y), 월(m), 일(d)을 정수로 변환
    time = tmp['Time']
    tmp['DTime'] ='-'.join(data_list.split('-')[-1].split('.')[:-1])
    ctime = time.apply(lambda _ : _.replace(u'오후', 'PM').replace(u'오전', 'AM'))
    n_time = ctime.apply(lambda _ : datetime.datetime.strptime(_, "%p %I:%M:%S.%f"))
    newtime = n_time.apply(lambda _ : _.replace(year=y, month=m, day=d))
    tmp['Time'] = newtime
    
    return tmp




# dd = csv_read_(root_dir, data_lists[0]) # 가이드북 코드하면 실행안돼. 내 생각엔 파씽에 적합하지 않은 형태

# 수정본
dd = csv_read_(root_dir, data_lists[1])

for i in range(2, len(data_lists)):
    dd = pd.merge(dd, csv_read_(root_dir, data_lists[i]), how='outer') # how='outer' : 합집합으로 merge. 없는 행은 NaN으로 채우기
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
# correlation = dedicated_data.corr() # 실행안돼. 수정

# 수정ver
correlation = dedicated_data.drop('DTime', axis=1).corr()

correlation

# 2-9. 데이터 상관관계 시각화
sns.heatmap(correlation, annot=True, fmt='.2f') # -> 매우 낮은 상관관계







# 3. 데이터 정제(전처리)
# 3-1. null값 제거
dedicated_data = dedicated_data.dropna()
dedicated_data






# 4. 알고리즘 선택
# method1) autokeras내 StructuredDataRegressor 딥러닝 모델
# AutoML이 적용된 keras 라이브러리. <AutoML> : 하이퍼 파라미터를 스스로 변형하여 최적화 시키는 머신러닝을 의미

# method2) sklearn내 Decision Tree 머신러닝 모델








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
# X_data = pd.DataFrame(columns={'pH','Temp','Current', 'Voltage', 'QC'}) # 이것도 가이드북 오류인듯

# 수정ver
X_data = pd.DataFrame(columns=['pH', 'Temp', 'Current', 'Voltage', 'QC'])


# =============================================================================
# 
# for d in d_lists:
#     for lot in lot_lists:
#         tmp = dd[(dd['DTime']==d)&(dd['Lot']==lot)]
#         tmp = tmp[['pH', 'Temp','Current', 'Voltage']]
#         error_df = error_drop[(error_drop['Date']==d)&((error_drop['LoT']==lot))]
#         len_error = len(error_df)
#         
#         if len_error>0:
#             trr = np.full((tmp['pH'].shape), 0)
#         else:
#             trr = np.full((tmp['pH'].shape), 1)
#         tmp['QC'] = trr
#         X_data = X_data.append(tmp)
# =============================================================================
    
# pandas의 append()는 버전 2.0.0 이후부터 사라진 기능임.
# 수정ver
# 두 개의 데이터프레임에서 필요한 부분만 합치는 코드
for d in d_lists:
    for lot in lot_lists:
        tmp = dd[(dd['DTime']==d)&(dd['Lot']==lot)]
        tmp = tmp[['pH', 'Temp','Current', 'Voltage']]
        error_df = error_drop[(error_drop['Date']==d)&((error_drop['LoT']==lot))]
        len_error = len(error_df)
        
        if len_error > 0:
            trr = np.full((tmp['pH'].shape), 0)
        else:
            trr = np.full((tmp['pH'].shape), 1)
        tmp['QC'] = trr
        X_data = pd.concat([X_data, tmp], ignore_index=True)

X_data = X_data.apply(pd.to_numeric) # pd.to_numeric() 함수 : 데이터프레임의 모든 값이 숫자로 변환






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
# -> result : 학습용 데이터는, 데이터 별 상관관계가 無



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
# -> result : 테스트용 데이터는, 데이터 별 상관관계가 無







# 6. 모델링
# 6-1. Regression 모델 모델링

is_training = True  # 현재 모드가 모델 훈련 모드임을 나타냄
if is_training:
    # reg = ak.StructuredDataRegressor(overwrite=True, max_trials=5) # 이 부분이 계속해서 오류 발생. 디렉토리 문제인 것 같은데, 명시적으로 설정해줘도 계속 문제 발생.
    reg = ak.StructuredDataRegressor(overwrite=True, max_trials=5, directory=model_dir)

    reg.fit(train_data[['pH','Temp','Current', 'Voltage']], train_data[['QC']], verbose=2, epochs=7)
    model = reg.export_model()
else:
    model = tensorflow.keras.models.load_model("structured_data_regressor/best_model", custom_objects=ak.CUSTOM_OBJECTS)




# =============================================================================
# 모델 훈련:
# 
# is_training이 True인 경우, StructuredDataRegressor를 사용하여 구조화된 데이터에 대한 회귀 모델을 생성
# 여기서 overwrite=True는 이전에 저장된 모델이나 트라이얼(trial)이 있다면 덮어쓰겠다는 의미이며, max_trials=5는 최대 5번의 시도를 통해 최적의 모델을 찾겠다는 것을 의미
# reg.fit 메서드를 호출하여 모델 훈련. 이때, train_data에서 독립 변수로 사용될 'pH', 'Temp', 'Current', 'Voltage' 열 = 인풋, 종속 변수 'QC' = 아웃풋
# verbose=2는 훈련 과정에서의 로그 출력 수준을 설정하고, epochs=7은 전체 데이터셋을 7회 반복 학습하겠다는 것을 의미
# reg.export_model() 메서드를 호출하여 훈련된 모델을 내보냄
# 이렇게 내보낸 모델은 TensorFlow의 Keras 모델 객체로, 추가 분석이나 예측, 저장 등에 사용 가능
# 모델 로드:
# 
# is_training이 False인 경우, 즉 모델 훈련 모드가 아닐 때는 tensorflow.keras.models.load_model 메서드를 사용하여 사전에 저장된 모델을 로드.
# 여기서 "structured_data_regressor/best_model"은 저장된 모델의 경로와 파일명을 나타냄.
# custom_objects=ak.CUSTOM_OBJECTS는 AutoKeras에서 정의된 사용자 정의 객체들을 로드하는 데 필요한 매개변수. 이는 AutoKeras가 훈련 과정에서 사용할 수 있는 특정 사용자 정의 층이나 함수 등을 포함할 수 있기 때문에 필요
# 이 코드는 AutoKeras를 사용하여 구조화된 데이터에 대한 회귀 문제를 해결하는 전체적인 과정을 보여줌. 모델 훈련과 로드를 조건에 따라 분기 처리하여, 훈련 모드와 평가/예측 모드를 구분.
# 
# =============================================================================








# 6-2. DecisionTree 모델 모델링
clf = tree.DecisionTreeRegressor()








# 7. 모델훈련
# 7-1. Regression 모델 학습
reg.fit(X_data[['pH','Temp','Current']], X_data[['QC']], verbose=2, epochs=10)


# 7-2. Regression 모델 시각화
model.summary()

plot_model(model)

# 7-3. Decision Tree 모델 학습
clf = clf.fit(X_data[['pH','Temp','Current', 'Voltage']], X_data[['QC']])

# 7-4. Decision Tree 모델 시각화
vis = True
if vis:
    plt.figure(figsize=(10,30))
    tree.plot_tree(clf)
    plt.show()
# 아주 복잡한 형상이 출력






# 8. 모델 튜닝
# 8-1. Regression 모델 튜닝
# Regression 모델의 경우 튜닝할 것이 max_trials로 정해져 있음.
# max_trials의 경우 모델 구조에 따라 달라질 수 있고, 최대 횟수 중 가장 좋은 모델을 최종적으로 선택 -> 많이 한다고 꼭 좋은건 아냐



# 8-2. Decision Tree 모델 튜닝
# max_depth를 3으로 해서 학습
new_clf = tree.DecisionTreeRegressor(max_depth=3)
new_clf = new_clf.fit(train_data[['pH','Temp','Current', 'Voltage']], train_data[['QC']])

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



# 9-2. Regression 모델 해석




# 9-3. Decision Tree 모델 평가
predicted_data = clf.predict(test_data[['pH','Temp','Current', 'Voltage']])
print('Decision Tree Model Predict : ', predicted_data)
rmse = sqrt(mean_squared_error(test_data['QC'], predicted_data))
print('Decision Tree Model RMSE : ',rmse)

# 정확도가 매우 높음



y_test = test_data['QC']
y_pred = [round(y, 0) for y in predicted_data]
print("accuracy = ", accuracy_score(y_test, y_pred))
print("recall = ", recall_score(y_test, y_pred))
print("precision = ", precision_score(y_test, y_pred))
print("f1 score = ", f1_score(y_test, y_pred))


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
ax1.scatter(test_data['pH'], test_data['Temp'], predicted_data)
ax1.set_xlabel('pH Test Data')
ax1.set_ylabel('Temp Test Data')
ax1.set_zlabel('Estimated Process Data')
ax2 = fig.add_subplot(3,2,2, projection='3d')
ax2.scatter(test_data['Temp'], test_data['Current'], predicted_data)
ax2.set_xlabel('Temp Test Data')
ax2.set_ylabel('Current Test Data')
ax2.set_zlabel('Estimated Process Data')
ax3 = fig.add_subplot(3,2,3, projection='3d')
ax3.scatter(test_data['Current'], test_data['Voltage'], predicted_data)
ax3.set_xlabel('pH Test Data')
ax3.set_ylabel('Voltage Test Data')
ax3.set_zlabel('Estimated Process Data')
ax4 = fig.add_subplot(3,2,4, projection='3d')
ax4.scatter(test_data['Voltage'], test_data['pH'], predicted_data)
ax4.set_xlabel('Voltage Test Data')
ax4.set_ylabel('pH Test Data')
ax4.set_zlabel('Estimated Process Data')
ax5 = fig.add_subplot(3,2,5, projection='3d')
ax5.scatter(test_data['pH'], test_data['Current'], predicted_data)
ax5.set_xlabel('pH Test Data')
ax5.set_ylabel('Current Test Data')
ax5.set_zlabel('Estimated Process Data')
ax6 = fig.add_subplot(3,2,6, projection='3d')
ax6.scatter(test_data['Temp'], test_data['Voltage'], predicted_data)
ax6.set_xlabel('Temp Test Data')
ax6.set_ylabel('Voltage Test Data')
ax6.set_zlabel('Estimated Process Data')
plt.tight_layout()
plt.show()



# 9-4. Decision Tree 모델 해석

# 9-5. Tuned Decision Tree 모델 평가
predicted_data = new_clf.predict(test_data[['pH','Temp','Current', 'Voltage']])
print('Decision Tree Model Predict : ', predicted_data)
rmse = sqrt(mean_squared_error(test_data['QC'], predicted_data))
print('Decision Tree Model RMSE : ',rmse)


y_test = test_data['QC'].values
y_pred = [round(y, 0) for y in predicted_data]
print("accuracy = ", accuracy_score(y_test, y_pred))
print("recall = ", recall_score(y_test, y_pred))
print("precision = ", precision_score(y_test, y_pred))
print("f1 score = ", f1_score(y_test, y_pred))



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
ax1.scatter(test_data['pH'], test_data['Temp'], predicted_data)
ax1.set_xlabel('pH Test Data')
ax1.set_ylabel('Temp Test Data')
ax1.set_zlabel('Estimated Process Data')
ax2 = fig.add_subplot(3,2,2, projection='3d')
ax2.scatter(test_data['Temp'], test_data['Current'], predicted_data)
ax2.set_xlabel('Temp Test Data')
ax2.set_ylabel('Current Test Data')
ax2.set_zlabel('Estimated Process Data')
ax3 = fig.add_subplot(3,2,3, projection='3d')
ax3.scatter(test_data['Current'], test_data['Voltage'], predicted_data)
ax3.set_xlabel('pH Test Data')
ax3.set_ylabel('Voltage Test Data')
ax3.set_zlabel('Estimated Process Data')
ax4 = fig.add_subplot(3,2,4, projection='3d')
ax4.scatter(test_data['Voltage'], test_data['pH'], predicted_data)
ax4.set_xlabel('Voltage Test Data')
ax4.set_ylabel('pH Test Data')
ax4.set_zlabel('Estimated Process Data')
ax5 = fig.add_subplot(3,2,5, projection='3d')
ax5.scatter(test_data['pH'], test_data['Current'], predicted_data)
ax5.set_xlabel('pH Test Data')
ax5.set_ylabel('Current Test Data')
ax5.set_zlabel('Estimated Process Data')
ax6 = fig.add_subplot(3,2,6, projection='3d')
ax6.scatter(test_data['Temp'], test_data['Voltage'], predicted_data)
ax6.set_xlabel('Temp Test Data')
ax6.set_ylabel('Voltage Test Data')
ax6.set_zlabel('Estimated Process Data')
plt.tight_layout()
plt.show()


# 9-6. Tuned Decision Tree 모델 해석