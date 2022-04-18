import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cnn_models.char_cnn import CharCNN
from sklearn import metrics
from util import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="char_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()


NUM_CLASS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 10
CHAR_MAX_LEN = 123

print("Building dataset...")
x, y, alphabet_size = build_char_dataset("train_64.pkl")

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)

with tf.compat.v1.Session() as sess:
    model = CharCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        total_pred = []
        total_act = []

        if step % 2000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                accuracy, predictions = sess.run([model.accuracy, model.predictions], feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
                total_pred += list(predictions)
                total_act += list(valid_y_batch)
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))
            print(metrics.confusion_matrix(total_act, total_pred))
            print(metrics.classification_report(total_act, total_pred))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")
