import tensorflow as tf

class naive_net:
	def __init__(self, training=False):
		# train / test switch for dropout
		self.is_training = training
		print('naive_net.is_training='+str(self.is_training))

		# define input of network
		self.input = tf.placeholder(tf.float32, [None, 64, 192, 2])

		# define output
		self.output = self.network(self.input)

		# define loss
		self.y_gt = tf.placeholder(tf.float32, [None, 1]) 
		self.loss = self.loss_function()

	def network(self, x):
		conv_1 = tf.layers.conv2d(inputs=x,
								  filters=32,
								  kernel_size=(3,3),
								  strides=(1,1),
								  padding='same',
								  activation=tf.nn.leaky_relu,
								  name="conv_1")
		mp_1 = tf.layers.max_pooling2d(conv_1,
									   pool_size=(2,2),
									   strides=(2,2),
									   padding='same',
									   name="mp_1")
		bn_1 = tf.layers.batch_normalization(mp_1,
											name='bn_1')
		conv_2 = tf.layers.conv2d(inputs=bn_1,
								  filters=64,
								  kernel_size=(3,3),
								  strides=(1,1),
								  padding='same',
								  activation=tf.nn.leaky_relu,
								  name="conv_2")
		mp_2 = tf.layers.max_pooling2d(conv_2,
									   pool_size=(2,2),
									   strides=(2,2),
									   padding='same',
									   name="mp_2")
		bn_2 = tf.layers.batch_normalization(mp_2,
											name='bn_2')
		conv_3 = tf.layers.conv2d(inputs=bn_2,
								  filters=96,
								  kernel_size=(3,3),
								  strides=(1,1),
								  padding='same',
								  activation=tf.nn.leaky_relu,
								  name="conv_3")
		mp_3 = tf.layers.max_pooling2d(conv_3,
									   pool_size=(2,2),
									   strides=(2,2),
									   padding='same',
									   name="mp_3")
		bn_3 = tf.layers.batch_normalization(mp_3,
											name='bn_3')
		conv_11 = tf.layers.conv2d(bn_3,
								  filters=16,
								  kernel_size=(1,1),
								  strides=(1,1),
								  padding='same',
								  activation=tf.nn.leaky_relu,
								  name="conv_11")
		bn_4 = tf.layers.batch_normalization(conv_11,
											name='bn_4')
		flatten = tf.layers.flatten(bn_4,
									name="flatten")
		dropout_1 = tf.layers.dropout(flatten,
										rate=0.3,
										training=self.is_training,
										name="dropout_1")
		fc_1 = tf.layers.dense(dropout_1, 
							   512, 
							   activation=tf.nn.leaky_relu,
							   name="fc_1")
		dropout_2 = tf.layers.dropout(fc_1,
										rate=0.3,
										training=self.is_training,
										name="dropout_2")
		fc_2 = tf.layers.dense(dropout_2, 
							   1, 
							   name="fc_2")
		sc = tf.multiply(fc_2, 12.0, name='sc')
		return sc

	def loss_function(self):
		# output format: [batch x 1]
		loss = tf.losses.mean_squared_error(labels=self.y_gt,
											predictions=self.output)
		return loss

if __name__ == '__main__':
	import numpy as np
	x = np.ones((8,64,192,3))
	y = np.ones((8,1))

	with tf.Session() as sess:
		net = naive_net(training=False)
		tf.global_variables_initializer().run()
		out = sess.run(net.output, feed_dict={net.input:x, net.y_gt:y})
		print(out)
		print(out.shape)