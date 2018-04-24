import tensorflow as tf
# 变量必须要定义为Variable才是变量
state = tf.Variable(0, name='counter')
# print(state.name)
# 常量必须要定义出常量才是常量
one = tf.constant(1)
# 变量与常量相加还是变量
new_value = tf.add(state, one)
# 将新变量的值赋予到原来这个变量上
update = tf.assign(state, new_value)
# 如果设置的变量，那么就需要初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        print(sess.run(update))
