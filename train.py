import os
from datetime import datetime
import time
import tensorflow as tf

from model import Model
import dataset_utils as du


# Training function. If retrain, replace pre_trained_model=None to the model name in the main fucntion.
# drop_rate: drop rate set for Dropout Regularization.
# filter_map: the number of filter used in convolutional layers.
# kernel_size_map: the size of kernel used in convolutional layers and maxpooling layer.
# learning rate
# dropout
# batch size

def my_training(ds, train_data, train_labels, val_data, val_labels,
                num_train, num_val, conv_featmap=[48,64,128,160,192], fc_units=[84], 
                conv_kernel_size=[[5,5],[2,2]], pooling_size=[2], l2_norm=0.015,
                learning_rate=1e-2, batch_size=32, decay =0.9, dropout=0.3, 
                verbose=False, pre_trained_model=None):
    print("Building my SVHN_CNN. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("learning_rate={}".format(learning_rate))
    #print("decay={}").format(decay)
    #print("dropout").format(dropout)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    with tf.Graph().as_default():
        #print (train_data.shape)
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 54, 54, 3], dtype=tf.float32)
            ys1 = tf.placeholder(shape=[None, ], dtype=tf.int32)
            ys2 = tf.placeholder(shape=[None, 5], dtype=tf.int32)
        length_logtis, digits_logits = Model.layers(xs, drop_rate=0.2)
        loss = Model.loss(length_logtis, digits_logits, ys1, ys2)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                   decay_steps=10000, decay_rate=decay, staircase=True)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        tf.summary.scalar('SVHN_loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        
        cur_model_name = 'SVHN_CNN_{}'.format(int(time.time()))

        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if pre_trained_model is not None:
                try: 

                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass
            

            print('Start training')
            init_tolerance = 100
            best_acc = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                image_batch, label = ds.build_batch(train_data, train_labels, batch_size, is_train=True, shuffle=True)
                length_batch = label[:, 0]
                digits_batch = label[:, 1:6]
                _, loss_train, summary_train, global_step_train, learning_rate_train = sess.run([optimizer, loss, merge, global_step, learning_rate], feed_dict={xs:image_batch, ys1:length_batch, ys2:digits_batch})
                duration += time.time() - start_time

                if global_step_train % 100 == 0:
                    
                    duration = 0.0
                    print('%s: iter_total %d, loss = %f' % (
                        datetime.now(), global_step_train, loss_train))

                if global_step_train % 1000 == 0:
                    

                    writer.add_summary(summary_train, global_step=global_step_train)


                    checkoutfile = saver.save(sess, os.path.join('model/', 'latest.ckpt'))
                    accuracy = evaluate(checkoutfile, ds, val_data, val_labels,
                                        
                                        num_val,
                                        global_step_train)
                    print('accuracy = %f' % (accuracy))

                    if accuracy > best_acc:
                        modelfile = saver.save(sess, os.path.join('model/', 'model.ckpt'),
                                                             global_step=global_step_train)
                        print('Best validation accuracy!' + modelfile)
                        tolerance = init_tolerance
                        best_acc = accuracy
                    else:
                        tolerance -= 1

                    print('remaining tolerance = %d' % tolerance)
                    if tolerance == 0:
                        break

            coord.request_stop()
            coord.join(threads)
            print("Traning ends. The best valid accuracy is {}.".format(best_acc))



##############################
#evaluation function
#evaluate on test dataset
##############################
def evaluate(path_to_checkpoint, ds, val_data, val_labels, num_examples, global_step):

    batch_size = 128
    num_batches = num_examples // batch_size
    needs_include_length = False

    with tf.Graph().as_default():
        with tf.name_scope('test_inputs'):
            xs = tf.placeholder(shape=[None, 54, 54, 3], dtype=tf.float32)
            ys1 = tf.placeholder(shape=[None, ], dtype=tf.int32)
            ys2 = tf.placeholder(shape=[None, 5], dtype=tf.int32)
        
        length_logits, digits_logits = Model.layers(xs, drop_rate=0.0)
        length_predictions = tf.argmax(length_logits, axis=1)
        digits_predictions = tf.argmax(digits_logits, axis=2)

        if needs_include_length:
            labels = tf.concat([tf.reshape(ys1, [-1, 1]), ys2], axis=1)
            predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
        else:
            labels = ys2
            predictions = digits_predictions

        labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
        predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

        accuracy, update_accuracy = tf.metrics.accuracy(
            labels=labels_string,
            predictions=predictions_string
        )

        tf.summary.image('image', xs)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('variables',
                             tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_checkpoint)

            for _ in range(num_batches):
                image_batch, label = ds.build_batch(val_data, val_labels, batch_size, is_train=False, shuffle=False)
                length_batch = label[:, 0]
                digits_batch = label[:, 1:6]
    
                acc, update = sess.run([accuracy, update_accuracy], feed_dict = {xs:image_batch, ys1:length_batch, ys2: digits_batch})
            coord.request_stop()
            coord.join(threads)

    return acc



def main(_):
    ds = du.dataset()
    train_data, test_data, train_labels, test_labels = ds.load_image([64,64])
    my_training(ds, train_data, train_labels, test_data, test_labels, 
                212052, 23702, conv_featmap=[48,64,128,160,192], fc_units=[84], 
                conv_kernel_size=[[5,5],[2,2]], pooling_size=[2], l2_norm=0.01,
                learning_rate=2e-2, batch_size=32, decay =0.9, dropout=0.15, 
                verbose=False, pre_trained_model=None)


if __name__ == '__main__':
    tf.app.run(main=main)
