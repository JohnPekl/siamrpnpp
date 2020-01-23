import tensorflow as tf
import numpy as np
from .gen_ancor import Anchor
from .anchor_tf import Anchor_tf
class Loss_op():
    def __init__(self, width, height):
        self.anchors=tf.convert_to_tensor(Anchor(width,height).anchors)
        self.anchor_tf=Anchor_tf()
    def loss(self,gt,pre_score,pre_box):
        label,target_box,target_inside_weight,target_outside_weight,all_box=self.anchor_tf.pos_neg_anchor2(gt,self.anchors)
        #=========cls_loss=============
        pre_score=tf.reshape(pre_score,(-1,2))
        pre_score_valid=tf.reshape(tf.gather(pre_score,tf.where(tf.not_equal(label,-1))),(-1,2))
        label_valid=tf.cast(tf.reshape(tf.gather(label,tf.where(tf.not_equal(label,-1))),[-1]),tf.int32)
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_score_valid, labels=label_valid))
        #=========reg_loss=============
        pre_box=tf.reshape(pre_box,(-1,4))
        inside=tf.multiply(target_inside_weight,tf.subtract(pre_box, target_box))
        mask=tf.cast(tf.less(tf.abs(inside),1),tf.float32)
        option1=tf.multiply(tf.multiply(inside,inside),0.5)
        option2=tf.subtract(tf.abs(inside),0.5)

        smooth_l1=tf.multiply(tf.add(tf.multiply(option1,mask),tf.multiply(option2,tf.subtract(1.,mask))),target_outside_weight)
        reg_loss=tf.reduce_sum(smooth_l1)
        return cls_loss,reg_loss,label,target_box





