# 11_3 카피

# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import pandas as pd                                                                        # csv파일을 데려오는 라이브러리
import time

# 1. 데이터
path = "./_data/"           # . : 현재 폴더

train_csv = pd.read_csv(path + "train.csv", index_col=0)            # "./_data/train.csv"
test_csv = pd.read_csv(path + "test.csv", index_col=0)                           # "./data/test.csv"
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.columns)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)             # pandas는 numpy 파일 형태
                                                                          # 열을 뺴려면 axis=1
                           
                                                                          
print("=============================================")
y = train_csv['count']
print(y)
print(y.shape)                  # (10886,)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,        # default
                                                    shuffle=True,            # default
                                                    random_state=100,
                                                    )

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(8,)))
model.add(Dropout(0.1))                                     # 10개의 노드중 10%를 노드에 쓰지 않음
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(190, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))                                 # linear(선형회귀식) : 그 값 그대로 쓰겠다


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
start_time = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=64)                   # batch_size default값 : 32
end_time = time.time()


loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                      # loss =  2601.837890625
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)                    # r2 score :  0.5394820038443388


###### csv 파일 만들기 ######
y_submit = model.predict(test_csv)
submit_csv['count'] = y_submit
submit_csv.to_csv(path + "submission_0717_1045.csv")

print("gpu 걸린시간 :",round(end_time - start_time, 2), "초")       # 소수 둘째 자리까지 반올림