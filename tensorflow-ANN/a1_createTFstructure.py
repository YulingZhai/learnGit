import tensorflow as tf
import numpy as np
# 创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3
# 定义Weights为矩阵，初始值Weights为一维的从-1到1之间的随机数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 定义biases为一个一维的0
biases = tf.Variable(tf.zeros([1]))
# y为根据自己的
y = Weights*x_data+biases
# 损失为预测数据和实际的数据的差值
loss = tf.reduce_mean(tf.square(y-y_data))
# 优化方法为梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练为最小化这个误差损失
train = optimizer.minimize(loss)
# 初始化所有的变量
init = tf.global_variables_initializer()
sess = tf.Session()
# 执行所有的初始化动作
sess.run(init)
print(sess.run(Weights), sess.run(biases))
for step in range(201):
    # 执行训练操作
    sess.run(train)
    if step % 20 == 0:
        # 查看此时的Weights,biases的值是多少
        print(step, sess.run(Weights), sess.run(biases))
