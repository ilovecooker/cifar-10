from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
class get_data():
    def get_cifar10_data(self):

        # x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
        (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

        # 验证集分配（从测试集中抽取，因为训练集数据量不够）
        x_val = x_test_original[:5000]
        y_val = y_test_original[:5000]
        x_test = x_test_original[5000:]
        y_test = y_test_original[5000:]
        x_train = x_train_original
        y_train = y_train_original

        # 这里把数据从unint类型转化为float32类型, 提高训练精度。
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，通常将数值归一化映射到0-1。
        x_train = x_train / 255
        x_val = x_val / 255
        x_test = x_test / 255

        # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_val, y_val, x_test, y_test
