from utills import model3, lr, _Save, _Test
import tensorflow as tf
from time import time
import math
from data import get_data_set
from dataInput import dataInput

if __name__ == '__main__': 
    train_x, train_y = get_data_set("train")
    test_x, test_y = get_data_set("test")
    
    dataVisual = dataInput()
    dataVisual.dataVisuallizationSubplot()

    x, y, output, y_pred_cls, global_step, learning_rate = model3()
    global_accuracy = 0

    # PARAMS
    _BATCH_SIZE = 256
    _EPOCH = 50
    _SAVE_PATH = "./ModelInfo/cifar-10-v1.0.0/"   
    
    # LOSS AND OPTIMIZER
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)
        
    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # SAVER
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)  
        
#    try:
#        print("\nTrying to restore last checkpoint ...")
#        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
#        saver.restore(sess, save_path=last_chk_path)
#        print("Restored checkpoint from:", last_chk_path)
#    except ValueError:
#        print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0    
    for epoch in range(_EPOCH):
        print(str(epoch + 1) + ' epoch')
        for s in range(batch_size):
            batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
            batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]            
            
            start_time = time()
            i_global, _, batch_loss, batch_acc = sess.run(
                    [global_step, optimizer, loss, accuracy],
                    feed_dict={x: batch_xs, y: batch_ys, learning_rate:lr(epoch)} ) 
            duration = time() - start_time        
            
            if s % 10 == 0:
                percentage = int(round((s/batch_size)*100))
                bar_len = 29
                filled_len = int((bar_len*int(percentage))/100)
                bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
                msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
                print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))            
        
        _Save(saver, sess, _SAVE_PATH, i_global)


    acc = _Test(x, y, test_x, test_y, sess, y_pred_cls)
    print('Test ACC Rate --> ' + str(acc) + ' %')
    print('main')