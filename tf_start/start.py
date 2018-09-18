#-*-conding:utf-8-*-
import tensorflow as tf
import numpy as np

x_data = np.float64(np.random.rand(2, 100))
y_data = np.dot([0.2, 0.3], x_data) + 0.5

with tf.name_scope('val') as scope:
  b = tf.Variable(np.zeros(1), name='bias')
  w = tf.Variable(np.random.uniform(-1, 1, (1, 2)), name='weights')
  y = tf.matmul(w, x_data) + b

loss = tf.reduce_mean(np.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

tf.summary.scalar('loss value', loss)

with tf.Session() as session:
  session.run(init)

  merged_summary_ops = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('./logs', session.graph)

  step = 0
  while step <= 400:
    step += 1
    session.run(train)
    if step % 10 == 0:
      print(step, session.run(w), session.run(b))
      summary_str = session.run(merged_summary_ops)
      summary_writer.add_summary(summary_str, step)
