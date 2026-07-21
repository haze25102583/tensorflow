# 14_1 카피
# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes                                                 # 교육용 데이터 제공
import pandas as pd                                                                        # csv파일을 데려오는 라이브러리
import time


# rainbow_csv, edit_csv
# 1. 데이터
path = "./_data/"           # . : 현재 폴더

train_csv = pd.read_csv(path + "train.csv", index_col=0)            # "./_data/train.csv"
test_csv = pd.read_csv(path + "test.csv", index_col=0)                           # "./data/test.csv"
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv)
# print(train_csv.shape)        # (10886, 11)

# print(test_csv)
# print(test_csv.shape)         # (6493, 8)

# print(submit_csv)
# print(submit_csv.shape)         # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='str')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)             # pandas는 numpy 파일 형태
                                                                          # 열을 뺴려면 axis=1
                           
                                                                          
# print(x)                                                                    

print("=============================================")
y = train_csv['count']
print(y)
print(y.shape)                  # (10886,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,        # default
                                                    shuffle=True,            # default
                                                    random_state=100,
                                                    )

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(8,)))
model.add(Dense(28, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1, activation='linear'))                                 # linear(선형회귀식) : 그 값 그대로 쓰겠다

# 3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',
    patience=100,                # 최소값이 갱신되지 않는 횟수
    verbose=1,
    restore_best_weights=True,
    )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose = 1,                 # 훈련도중, 저장을 보여줌
    save_best_only=True,
    filepath='./_save/keras23_mcp1.keras'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100000000000000, batch_size=64,
          validation_split=0.2,
          callbacks = [es, mcp])                   # EarlyStopping 적용
                                                   # [] 형태 -> callbacks에 이후 들어가는 내용'들'이 있기 때문
                                                                                               # validation_split
                                                                                               # 전체 말고 training data의 0.2
                                                                                               # val_loss : 
end_time = time.time()


# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                      # loss =  2601.837890625
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)                    # r2 score :  0.5394820038443388


###### csv 파일 만들기 ######
y_submit = model.predict(test_csv)
# print(y_submit)
submit_csv['count'] = y_submit
# print(submit_csv)
submit_csv.to_csv(path + "submission_0717_1453.csv")

print("걸린시간 :",round(end_time - start_time, 2), "초")       # 소수 둘째 자리까지 반올림