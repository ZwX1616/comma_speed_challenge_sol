import tensorflow as tf

class naive_net:
	def __init__(self, training=False):
		# train / test switch for dropout
		self.is_training = training
		print('naive_net.is_training='+str(self.is_training))

		# define input of network
		self.input = tf.placeholder(tf.float32, [None, 80, 240, 3])

		# define output
		self.output = self.network(self.input)

		# define loss
		self.y_gt = tf.placeholder(tf.float32, [None, 1]) 
		self.loss = self.loss_function()

	def network(self, x):
		conv_11 = tf.layers.conv2d(inputs=x,
								  filters=1,
								  kernel_size=(1,1),
								  strides=(1,1),
								  padding='same',
								  activation=tf.nn.leaky_relu,
								  name="conv_11")
		flatten = tf.layers.flatten(conv_11,
									name="flatten")
		dropout_1 = tf.layers.dropout(flatten,
										rate=0.2,
										training=self.is_training,
										name="dropout_1")
		fc_1 = tf.layers.dense(dropout_1, 
							   512, 
							   activation=tf.nn.leaky_relu,
							   name="fc_1")
		bn_1 = tf.layers.batch_normalization(fc_1,
											name='bn_1')
		fc_2 = tf.layers.dense(bn_1, 
							   1, 
							   name="fc_2")
		return fc_2

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