import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")               # 실험적인
print(gpus)                             # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
                                        # 0 : gpu를 찾음 -> 4장 붙인 카드의 0번째

if(gpus):
    print("GPU 돈다~")
else:
    print("GPU 없다~")      # 처음 에러 -> 가상환경 활성화x -> visual studio 하단
    
    """A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

-> numpy 버전이 tensorflow의 버전과 맞지 않음. numpy를 다운그레이드
    """