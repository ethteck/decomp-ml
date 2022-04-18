import tensorflow as tf
from util import *

BATCH_SIZE = 128

test_x, test_y, alphabet_size = build_char_dataset("train.pkl")

checkpoint_file = tf.train.latest_checkpoint("char_cnn")
graph = tf.Graph()
with graph.as_default():
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        predictions = graph.get_operation_by_name("fc-3/ArgMax").outputs[0]

        batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
        sum_accuracy, cnt = 0, 0
        for batch_x, batch_y in batches:
            feed_dict = {
                x: batch_x,
                y: batch_y,
                is_training: False
            }

            accuracy_out, predictions_out = sess.run([accuracy, predictions], feed_dict=feed_dict)
            sum_accuracy += accuracy_out
            cnt += 1

        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
