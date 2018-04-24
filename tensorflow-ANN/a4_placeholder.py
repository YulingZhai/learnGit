import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 喂养的字典，字典的内容为键值对，值的内容可以为列表
    print(sess.run(output, feed_dict={input1: 7., input2: [2.,1.]}))
