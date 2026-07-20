import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")               # 실험적인
print(gpus)

if(gpus):
    print("GPU 돈다~")
else:
    print("GPU 없다~")      # 처음 에러 -> 가상환경 활성화x -> visual studio 하단