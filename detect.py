from tensorflow.keras import models
from data_get import get_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()
data=get_data()
x_train, y_train, x_val, y_val, x_test, y_test = data.get_cifar10_data()
model=models.load_model("Lenet_cifar10.h5")
# 输出网络在测试集上的损失与精度
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 输出网络在测试集上的损失与精度
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 测试集结果预测
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
print('前20张图片预测结果：', predictions[:20])
# 预测结果图像可视化

def cifar10_visualize_multiple_predict(start, end, length, width):

    for i in range(start, end):
        plt.subplot(length, width, 1 + i)
        plt.imshow(x_test_original[5000+i], cmap=plt.get_cmap('gray'))
        title_true = 'true=' + str(y_test_original[5000+i])                  # 图像真实标签
        title_prediction = ',' + 'prediction' + str(predictions[i])     # 预测结果
        title = title_true + title_prediction
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()

cifar10_visualize_multiple_predict(start=0, end=9, length=3, width=3)
