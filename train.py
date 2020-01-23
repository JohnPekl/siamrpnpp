import tensorflow as tf
from modules.siamrpn import siamrpn
from reuse.utils.image_reader_vot2018 import Image_reader
from reuse.module.loss_module import Loss_op
import os
import numpy as np
import time
from reuse.module.gen_ancor import Anchor
from reuse.utils.debug import debug

reader = Image_reader('D://temp//dataset//_unpack//vot2018')
step_num = reader.img_num * 50
save_per_epoch = reader.img_num
width, height = 25, 25
loss_op = Loss_op(width, height)
learning_rate = 0.01
decay_rate = 0.95
decay_step = int(save_per_epoch/4)
model_dir = './checkpoint'
anchor_op = Anchor(width, height)
is_debug = True

def train():
    template = tf.keras.Input(shape=[127, 127, 3], dtype=tf.float32, batch_size = 1)
    detection = tf.keras.Input(shape=[255, 255, 3], dtype=tf.float32, batch_size = 1)
    gt_box = tf.placeholder(tf.float32, shape = [4])
    pre_reg, pre_cls = siamrpn(template, detection)

    cls_loss,reg_loss,label,target_box=loss_op.loss(gt_box,pre_cls,pre_reg)
    loss = cls_loss + 5 * reg_loss
    
    saver = tf.train.Saver(max_to_keep=50)
    global_step = tf.Variable(0,trainable = False)
    lr = tf.train.exponential_decay(0.001, global_step, decay_step, decay_rate, staircase = True)
    train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss,global_step)
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, update_ops])

    coord = tf.train.Coordinator()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    sess.run(tf.global_variables_initializer())
    epoch = 19
    t = time.time()

    # +++++++++++++++++++++debug++++++++++++++++++++++++++++++
    debug_pre_cls = tf.nn.softmax(pre_cls)
    debug_pre_reg = pre_reg
    debug_pre_score = tf.nn.softmax(tf.reshape(pre_cls, (-1, 2)))
    debug_pre_box = tf.reshape(pre_reg, (-1, 4))
    # +++++++++++++++++++++debug++++++++++++++++++++++++++++++

    for step in range(step_num):
        template1, _, detection1, gt_box1, offset, ratio, detection_org, detection_label = reader.get_data()
        templatekk = np.ones((1, 127, 127, 3), np.float32)
        detectionkk = np.ones((1, 255, 255, 3), np.float32)
        templatekk[0]= template1
        detectionkk[0] = detection1
        
        lr_ = sess.run(lr)
        feed = {template: templatekk, detection: detectionkk}
        pre_reg_, pre_cls_ = sess.run([pre_reg, pre_cls], feed_dict = feed)

        feed = {pre_reg : pre_reg_, pre_cls : pre_cls_, gt_box : gt_box1}
        cls_loss_,reg_loss_,label_,target_box_ = sess.run([cls_loss,reg_loss, label,target_box], feed_dict = feed)
        

        feed = {cls_loss: cls_loss_, reg_loss: reg_loss_}
        loss_ = sess.run(loss, feed_dict=feed)
        feed = {loss: loss_, lr: lr_, template: templatekk, detection: detectionkk, gt_box: gt_box1}
        sess.run(train_op, feed_dict=feed)

        if is_debug and step > 1 and step % 1000 == 0:
            feed = {pre_reg: pre_reg_, pre_cls: pre_cls_, gt_box: gt_box1}
            debug_pre_cls_, debug_pre_reg_, debug_pre_score_, debug_pre_box_ = sess.run([debug_pre_cls, debug_pre_reg, debug_pre_score, debug_pre_box],
                                                                 feed_dict=feed)
            debug(detection1, gt_box1, debug_pre_cls_, debug_pre_reg_, debug_pre_score_,
                  debug_pre_box_, label_, target_box_, step + 7582000, anchor_op)

        if step %100 == 0:
            print('step={},loss={},cls_loss={},reg_loss={},lr={},time={}'.format(step,loss_,cls_loss_,reg_loss_,lr_,time.time()-t))
            t=time.time()
        if step %1000 == 0 and step > 0:
            epoch += 1
            save(saver, sess, model_dir, epoch)
    coord.request_stop()
    coord.join(threads)

def save(saver, sess, ckpt_path, epoch):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    model_path=os.path.join(ckpt_path,'model')
    saver.save(sess,model_path,epoch)
    print('saved model')

if __name__=='__main__':
    train()