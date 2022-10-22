import tensorflow as tf
from model import model_set
from data_get import get_data
import matplotlib.pyplot as plt
"""
编译网络并训练
"""
data=get_data()
x_train, y_train, x_val, y_val, x_test, y_test = data.get_cifar10_data()
model_fun=model_set()
model = model_fun.alexnet()

# 编译网络（定义损失函数、优化器、评估指标）
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#设置终止条件
early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.005,
                              patience=7, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)


# 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=100, verbose=1,callbacks = [early_stopping])
# 模型保存
model.save('alexnet_cifar10.h5')

# 定义训练过程可视化函数（训练集损失、验证集损失、训练集精度、验证集精度）
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')
