
# =============================================================================
# Date : 2024. 02. 22
# Author : 배태겸
# Subject : 사출성형 공급망최적화
#  
# =============================================================================





# 1. 라이브러리/데이터 불러오기
# 1-1. 패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from numpy import newaxis
import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', 15)


# 1-2. 데이터 불러오기
df = pd.read_csv('data/사출성형.csv') # 상대경로 사용 예시
df.head()




# 2. 데이터 탐색
# 2-1. 탐색할 데이터 선택
Product_86 = df[(df["Product_Number"]=="Product_86")]
Product_86.reset_index(drop=True, inplace=True)
Product_86.head()


# 2-2. 기초 통계 확인
Product_86.describe()


# 2-3. 데이터 시각화
plt.plot(Product_86[['T일 예정 수주량']])
plt.xlabel('index')
plt.ylabel('amount')




# 3. 데이터 정제(전처리)
# 3-1. 데이터 전처리

# 3-2. 데이터 정제
df.isna()

df.isna().sum()

# 3-3. 데이터 열 추가 및 제거
df.columns

df["Date"] = pd.to_datetime(df['DateTime']).dt.date
df["Time"] = pd.to_datetime(df['DateTime']).dt.time

select_col_df = df.drop(['DateTime'], axis=1)
select_col_df.columns

select_col_df = df.loc[:, ['Product_Number', 'T일 예정 수주량', 'T+1일 예정 수주량', 'T+2일 예정 수주량', 'T+3일 예정 수주량',
                           'T+4일 예정 수주량', '작년 T일 예정 수주량', '작년 T+1일 예정 수주량', '작년 T+2일 예정 수주량', '작년 T+3일 예정 수주량',
                           '작년 T+4일 예정 수주량', 'T일 예상 수주량', 'T+1일 예상 수주량', 'T+2일 예상 수주량', 'T+3일 예상 수주량', 'T+4일 예상 수주량',
                           'DoW', 'Temperature', 'Humidity', 'Date', 'Time']]
select_col_df.columns



# 3-4. 불필요한 행 제거 (하루에 여러 번 수집된 데이터)
select_col_df.sort_values(by=['Product_Number', 'Date', 'Time'], inplace=True)
drop_df = select_col_df.drop_duplicates(['Product_Number', 'Date'], keep='last')
drop_df.reset_index(drop=True, inplace=True)
drop_df.head()



# 3-5. 불필요한 행 제거 (95일 연속 수집되지 않은 데이터)
product_list = drop_df["Product_Number"].unique()
drop_list = []
for product in product_list:
    if drop_df["Product_Number"].value_counts()[product] != 95:
        drop_list.append(product)
drop_list


drop_idx = drop_df[drop_df["Product_Number"].isin(drop_list)].index
drop_df.drop(drop_idx, inplace=True)
drop_df.reset_index(drop=True, inplace=True)
drop_df["Product_Number"].value_counts()

prev_product_num = len(product_list)
curr_product_num = len(drop_df["Product_Number"].unique())
print(prev_product_num, curr_product_num)


# 4. 주요 변수 선택
# 4-1. 변수별 상관관계 파악을 위한 데이터 재배열
time_series_df = drop_df.copy()
time_series_columns = time_series_df.columns
for i in range(15):
    column_name = time_series_columns[i+1]
    time_series_df[column_name] = time_series_df[column_name].shift(i%5)
for i in range(4):
    time_series_df = time_series_df[time_series_df["Product_Number"].duplicated()]


time_series_df.reset_index(drop=True, inplace=True)
time_series_new_columns = ['Product_Number', 'T일 예정 수주량', 'T-1일 예정 수주량', 'T-2일 예정 수주량', 'T-3일예정 수주량', 'T-4일 예정 수주량',
                           '작년 T일 예정 수주량', '작년 T-1일 예정 수주량', '작년 T-2일 예정 수주량', '작년 T-3일 예정 수주량', '작년 T-4일 예정수주량',
                           'T일 예상 수주량', 'T-1일 예상 수주량', 'T-2일 예상 수주량', 'T-3일 예상 수주량', 'T-4일 예상 수주량', 'DoW',
                           'Temperature', 'Humidity', 'Date', 'Time']
time_series_df.columns = time_series_new_columns
time_series_df.head()


# 4-2. 변수별 상관관계 파악 및 현장의 도메인 지식을 기반으로 한 유의미한 변수 선택
time_series_df.corr().head(1)


# 4-3. 주요 변수 선택 및 분석을 위한 데이터 선정
time_series_df = time_series_df.loc[:, ['Product_Number','T일 예정 수주량', 'T-1일 예정 수주량', 'T-2일 예정 수주량',
                                        'T-3일 예정 수주량', 'T-4일 예정 수주량', '작년 T일 예정 수주량', '작년 T-1일 예정 수주량', '작년 T-2일 예정 수주량',
                                        '작년T-3일 예정 수주량', '작년 T-4일 예정 수주량', 'T일 예상 수주량', 'T-1일 예상 수주량', 'T-2일 예상 수주량',
                                        'T-3일 예상 수주량', 'T-4일 예상 수주량']].reset_index(drop=True)
time_series_df.head()



# 5. 모델의 입력 데이터 형태 설정
# 5-1. 데이터의 독립변수 및 종속변수 정의
time_series_df.drop(['Product_Number'],axis=1, inplace=True)
time_series_df

X = time_series_df.copy() # 독립변수
y = X.pop('T일 예정 수주량') # 종속변수
X
print(y)


# 6. 학습, 검증, 평가 데이터셋 분리 및 스케일링
# 6-1. 학습, 검증, 평가용 데이터셋 분리
# 데이터셋 분리, train:validation:test = 70:15:15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# 분리 이후 데이터 형태
print("X 학습: {}, X 검증: {}, X 평가: {}".format(X_train.shape, X_valid.shape, X_test.shape))
print("y 학습: {}, y 검증: {}, y 평가: {}".format(y_train.shape, y_valid.shape, y_test.shape))

# 6-2. 데이터 스케일링
# 전처리시 필요한 패키지 불러오기
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_valid_sc = scaler.transform(X_valid)

X_test_sc = scaler.transform(X_test)


# 6-3. 데이터 입력 차원 변환
X_train_sc = X_train_sc[..., newaxis]
X_valid_sc = X_valid_sc[..., newaxis]
X_test_sc = X_test_sc[..., newaxis]
print(X_train_sc.shape, X_valid_sc.shape, X_test_sc.shape)



# 7. 모델 구축 및 학습
# 7-1. 분석모델 구축
# 필요한 패키지 불러오기
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(14,1), activation='relu')) # (samples,
timesteps, features)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

# 7-2. 모델 컴파일
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim, loss='mae')

# 7-3. 모델 학습
history = model.fit(X_train_sc, y_train, epochs=50, batch_size=64, validation_data=(X_valid_sc, y_valid))

# 7-4. 학습 결과 확인
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = history.epoch
plt.figure(figsize=(10,5))
plt.plot(epoch, loss, 'r', label='Training loss')
plt.plot(epoch, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()


# 8. 모델 성능 확인
# 8-1. 평가용 데이터셋에 대한 예측 성능 확인
model.evaluate(X_test_sc, y_test) # MAE

# 8-2. 예측값 확인
y_pred = model.predict(X_test_sc)
y_pred



# 9. 분석결과에 대한 논의 및 해석
# 9-1. 분석 결과 해석 가이드
plt.figure(figsize=(15,5))
plt.plot(range(len(y_test)), y_test, 'k.-', label="test")
plt.plot(range(len(y_pred)), y_pred, 'g.-', label="pred")
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(range(100), y_test[:100], 'k.-',label="test")
plt.plot(range(100), y_pred[:100], 'g.-',label="pred")
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(range(200,300), y_test[200:300], 'k.-',label="test")
plt.plot(range(200,300), y_pred[200:300], 'g.-',label="pred")
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(range(200,300), y_test[200:300], 'k.-',label="test")
plt.plot(range(200,300), y_pred[200:300], 'g.-',label="pred")
plt.legend()
plt.show()

# 9-2. 분석 결과 활용 가이드






