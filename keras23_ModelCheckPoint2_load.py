# 23_1 카피
# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
from tensorflow.keras.models import Sequential, load_model
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

model = load_model('./_save/keras23_mcp1.keras')

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
