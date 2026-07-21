from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes                                # 교육용 데이터 제공

# 1. 데이터
datasets = load_diabetes()
# print(datasets)
# print(datasets.DESCR)                             # describe : 조사하다
# # dataset : 행
# # attribute : 열
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# exit()
# x, y의 데이터가 분리
x = datasets.data
y = datasets.target
# print(x)
# print(y)

print(x.shape, y.shape)                 # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,        # default
                                                    shuffle=True,            # default
                                                    random_state=100,
                                                    )

print(x_train.shape, x_test.shape)  # (331, 10) (111, 10)
print(y_train.shape, y_test.shape)  # (331,) (111,)
# exit()



# 2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_shape=(10,)))
# model.add(Dense(28))
# model.add(Dense(50))
# model.add(Dense(19))
# model.add(Dense(1))

# model.summary()

# model.save("./_save/keras22_1_save_model.keras")

model = load_model("./_save/keras22_1_save_model.keras")
model.summary()

# exit()
# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=128, batch_size=64)                   # batch_size default값 : 32


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                      # loss =  2601.837890625
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)                    # r2 score :  0.5394820038443388

# R2 Score = 0.6 이상 ~ 0.9
