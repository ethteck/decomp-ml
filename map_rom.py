import tensorflow as tf
from util import *

BATCH_SIZE = 128

checkpoint_file = tf.train.latest_checkpoint("char_cnn")
graph = tf.Graph()

chunk_size = 64
with open("drmario.z64", "rb") as f:
    rom_bytes = f.read()


final_preds = []

with graph.as_default():
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        predictions = graph.get_operation_by_name("fc-3/ArgMax").outputs[0]

        curr_pred = -1
        section_start = 0

        for i in range(0, 0xB00000, chunk_size):
            seg_bytes = rom_bytes[i:i + chunk_size]
            x_list = list(seg_bytes) + [0] * (123 - len(seg_bytes))
            feed_dict = {
                x: [x_list],
                y: [0] * 123,
                is_training: False
            }

            predictions_out = sess.run([predictions], feed_dict=feed_dict)

            this_pred = predictions_out[0][0]
            if this_pred != curr_pred:
                final_preds.append((section_start, i, curr_pred))
                # print(hex(section_start) + " - " + hex(i) + ": " + ("data" if curr_pred == 0 else "code"))
                section_start = i
                curr_pred = this_pred
            dog = 5

            if i % 0x10000 == 0:
                print("Scanned " + hex(i) + " bytes")

print(str(len(final_preds)) + " preds")
with open("drmario_preds.pkl", "wb") as f:
    pickle.dump(final_preds, f, pickle.HIGHEST_PROTOCOL)
