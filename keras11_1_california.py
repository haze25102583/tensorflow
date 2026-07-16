from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing                                # 교육용 데이터 제공

# 1. 데이터
datasets = fetch_california_housing()
# print(datasets)
# print(datasets.DESCR)                             # describe : 조사하다
# # dataset : 행
# # attribute : 열
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
#  'Population', 'AveOccup', 'Latitude', 'Longitude']

# x, y의 데이터가 분리
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape, y.shape)                 # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,        # default
                                                    shuffle=True,            # default
                                                    random_state=103,
                                                    )

print(x_train.shape, x_test.shape)  # (20640, 8) (20640,)
print(y_train.shape, y_test.shape)  # (15480, 8) (5160, 8)



# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape=(8,)))
model.add(Dense(30))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=50, batch_size=64)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                      # loss =  0.6473488211631775

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)                    # r2 score :  0.5146815942363077

# R2 Score = 0.6 이상 ~ 0.9
