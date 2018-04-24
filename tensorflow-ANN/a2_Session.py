import tensorflow as tf
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
# 执行矩阵相乘
product = tf.matmul(matrix1, matrix2)
sess = tf.Session()
# 使用sess来运行这个相乘的操作
result = sess.run(product)
print(result)
sess.close()