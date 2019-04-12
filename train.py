import tensorflow as tf
import numpy as np
from time import time

from network import naive_net
from dataloader import DataLoader_train

learning_rate = 0.00001
num_epoch = 100
batch_size = 128

save_interval = 1
load_previous = True

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

net = naive_net(training=True)

# variable saver
saver = tf.train.Saver()

if load_previous: 
	saver.restore(sess, './checkpoints/model')
	with open('./checkpoints/iter.txt','r+') as f:
		epoch_offset = int(f.read())+1
	print ('...checkpoint loaded from epoch ' + \
			str(epoch_offset) + \
			'...')
else:
	tf.global_variables_initializer().run()
	epoch_offset = 0
	print ('...training from scratch')


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=net.loss)

my_dataloader = DataLoader_train(batch_size)

# tensorboard logger
writer_train = tf.summary.FileWriter('./tensorboard/')
tf.summary.scalar("MSE_loss", net.loss)
merged_summary = tf.summary.merge_all()

# iterate
for epoch in range(num_epoch):
	start = time()
	while (my_dataloader.is_complete==False):
		x, y = my_dataloader.get_batch()
		_, summary, l= sess.run([train_step, merged_summary, net.loss], feed_dict={net.input:x, net.y_gt:y})
		# import pdb; pdb.set_trace() ###
		if np.isnan(l):
			print('Model diverged with loss = NaN')
			quit()

		# logging
		print ('batch %d: loss %.5f, runtime: %.1fs' % (my_dataloader.current_batch-1, l, time()-start))

	print('epoch time: '+str(time()-start))
	if (epoch+epoch_offset==0):
		writer_train.add_graph(sess.graph)
	writer_train.add_summary(summary, epoch+epoch_offset)
	writer_train.flush()
	if ((epoch+epoch_offset)>0 and (epoch+epoch_offset)%save_interval==0):
		saver.save(sess, './checkpoints/model')
		with open('./checkpoints/iter.txt','w+') as f:
			f.write(str(epoch+epoch_offset))
		print ('...checkpoint saved...')
	my_dataloader.reset_epoch()

print (str(num_epoch)+' epochs completed')
writer_train.close()
sess.close()

# tensorboard --logdir="C:\Users\weixing\Documents\code\OA\comma_speed_challenge\tensorboard" --host=127.0.0.1