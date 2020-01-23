
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.backend import expand_dims

from .resnet50_rpn import resnet50

def padding_input(input, padding):
    pad_const = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
    x_pad = tf.pad(input, pad_const, "CONSTANT")
    return x_pad

# suppose examplar 127x127, search_region 255x255 are inputs of Resnet50
# examplar 15x15, output of search_region 31x31 (output of conv3, conv4, conv5 of Resnet50)
# carefully check: "Input (batch, height, width, channels)", 18 is batch size
def rpn(examplar, search_region):    
    # crop the center 7x7 regions, 
    examplar_crop = tf.image.resize_with_crop_or_pad(examplar, 7, 7)
    # adj_1 use for depth-wise cross correlation with filters: [filter_height, filter_width, in_channels, out_channels]
    adj_1 = Conv2D(filters = 256, kernel_size = 3, strides = 1) (padding_input(examplar_crop, 1))
    adj_1 = BatchNormalization() (adj_1)             # [18, 256, 7, 7]
    adj_1 = tf.reshape(adj_1, [-1, 7, 7])            # [4608, 7, 7]
    adj_1 = expand_dims(adj_1, axis = 0)             # [1, 4608, 7, 7]
    adj_1 = tf.transpose(adj_1, perm = (2, 3, 0, 1)) # [7, 7, 1, 4608] 
    
    adj_2 = Conv2D(filters = 256, kernel_size = 3, strides = 1) (padding_input(search_region, 1))
    adj_2 = BatchNormalization() (adj_2)             # [18, 256, 31, 31]
    adj_2 = tf.reshape(adj_2, [-1, 31, 31])          # [4608, 31, 31]
    adj_2 = expand_dims(adj_2, axis = 0)             # [1, 4608, 31, 31]
    adj_2 = tf.transpose(adj_2, perm = (0, 2, 3, 1)) # [1, 31, 31, 4608]
    
    # adj_3 use for depth-wise cross correlation with filters: [filter_height, filter_width, in_channels, out_channels]
    adj_3 = Conv2D(filters = 256, kernel_size = 3, strides = 1) (padding_input(examplar_crop, 1))
    adj_3 = BatchNormalization() (adj_3)
    adj_3 = tf.reshape(adj_3, [-1, 7, 7])
    adj_3 = expand_dims(adj_3, axis = 0)
    adj_3 = tf.transpose(adj_3, perm = (2, 3, 0, 1)) # [7, 7, 1, 4608]
    
    adj_4 = Conv2D(filters = 256, kernel_size = 3, strides = 1) (padding_input(search_region, 1))
    adj_4 = BatchNormalization() (adj_4)
    adj_4 = tf.reshape(adj_4, [-1, 31, 31])
    adj_4 = expand_dims(adj_4, axis = 0)
    adj_4 = tf.transpose(adj_4, perm = (0, 2, 3, 1)) # [1, 31, 31, 4608]
    
    # depth-wise cross correlation
    # input: [batch, in_height, in_width, in_channels]
    # filters: [filter_height, filter_width, in_channels, out_channels]
    dw_cross_region = tf.nn.conv2d(input = adj_2, filters = adj_1, strides = 1, padding = "VALID")
    dw_cross_region = tf.squeeze(input = dw_cross_region, axis = 0)  # [25, 25, 4608]
    dw_cross_region = tf.reshape(dw_cross_region, [-1, 25, 25, 256]) # [18, 25, 25, 256]
    
    dw_cross_class = tf.nn.conv2d(input = adj_4, filters = adj_3, strides = 1, padding = "VALID")
    dw_cross_class = tf.squeeze(input = dw_cross_class, axis = 0)
    dw_cross_class = tf.reshape(dw_cross_class, [-1, 25, 25, 256])
    
    # fusion
    dw_cross_region = Conv2D(filters = 256, kernel_size = 1, strides = 1) (dw_cross_region)
    dw_cross_region = BatchNormalization() (dw_cross_region)
    
    dw_cross_class = Conv2D(filters = 256, kernel_size = 1, strides = 1) (dw_cross_class)
    dw_cross_class = BatchNormalization() (dw_cross_class)
    
    # head
    bbox_prediction = Conv2D(filters = 4 * 5, kernel_size = 1, strides = 1) (dw_cross_region) # [18, 25, 25, 20] (4*k)
    class_prediction = Conv2D(filters = 2 * 5, kernel_size = 1, strides = 1) (dw_cross_class)  # [18, 25, 25, 10] (2*k)
    
    return bbox_prediction, class_prediction

## Test
#import numpy as np #4D tensor with shape: (batch, rows, cols, channels)
#examplar_feature_map = np.zeros(shape = [18, 15, 15, 256], dtype=np.float32)
#search_region_feature_map = np.zeros(shape = [18, 31, 31, 256], dtype=np.float32)
#output = rpn(examplar_feature_map, search_region_feature_map)

def siamrpn(examplar, search_region):
    examplar_conv3, examplar_conv4, examplar_conv5 = resnet50(examplar) # All [18, 15, 15, 256]
    searchregion_conv3, searchregion_conv4, searchregion_conv5 = resnet50(search_region) # All [18, 31, 31, 256]
    # feed into RPN module, output bbox [18, 25, 25, 20] (4*k), class [18, 25, 25, 10] (2*k)
    bbox_predt_conv3, cls_predt_conv3 = rpn(examplar_conv3, searchregion_conv3)
    bbox_predt_conv4, cls_predt_conv4 = rpn(examplar_conv4, searchregion_conv4)
    bbox_predt_conv5, cls_predt_conv5 = rpn(examplar_conv5, searchregion_conv5)
    # stack output
    batch_size = examplar.shape[0]
    cls_concat = tf.concat((cls_predt_conv3, cls_predt_conv4, cls_predt_conv5), 1) # [18, 75, 25, 10]
    cls_concat = tf.reshape(cls_concat, [batch_size, 25, 25, 10, -1])              # [18, 25, 25, 10, 3]
    cls_predt_stack = tf.reshape(cls_concat, [batch_size, 25, 25, -1])             # [18, 25, 25, 30]
    
    bbox_concat = tf.concat((bbox_predt_conv3, bbox_predt_conv4, bbox_predt_conv5), 1) # [18, 75, 25, 20]
    bbox_concat = tf.reshape(bbox_concat, [batch_size, 25, 25, 20, -1])                # [18, 25, 25, 20, 3]
    bbox_predt_stack = tf.reshape(bbox_concat, [batch_size, 25, 25, -1])               # [18, 25, 25, 60]
    # weight-fusion combines all outputs
    fused_cls_predt = Conv2D(filters = 10, kernel_size = 1, strides = 1) (cls_predt_stack)  # [18, 25, 25, 10]
    fused_bbox_predt = Conv2D(filters = 20, kernel_size = 1, strides = 1) (bbox_predt_stack)# [18, 25, 25, 20]
    
    return fused_bbox_predt, fused_cls_predt

## Test
#import numpy as np #4D tensor with shape: (batch, rows, cols, channels)
#examplar = np.zeros(shape = [18, 127, 127, 3], dtype=np.float32)
#search_region = np.zeros(shape = [18, 255, 255, 3], dtype=np.float32)
#output = siamrpn(examplar, search_region)

def siamrpn_conv(examplar_conv, searchregion_conv, batch_size):
    #examplar_conv3, examplar_conv4, examplar_conv5 = resnet50(examplar)  # All [18, 15, 15, 256]
    #searchregion_conv3, searchregion_conv4, searchregion_conv5 = resnet50(search_region)  # All [18, 31, 31, 256]
    # feed into RPN module, output bbox [18, 25, 25, 20] (4*k), class [18, 25, 25, 10] (2*k)
    bbox_predt_conv3, cls_predt_conv3 = rpn(examplar_conv[0], searchregion_conv[0])
    bbox_predt_conv4, cls_predt_conv4 = rpn(examplar_conv[1], searchregion_conv[1])
    bbox_predt_conv5, cls_predt_conv5 = rpn(examplar_conv[2], searchregion_conv[2])
    # stack output
    #batch_size = search_region.shape[0]
    cls_concat = tf.concat((cls_predt_conv3, cls_predt_conv4, cls_predt_conv5), 1)  # [18, 75, 25, 10]
    cls_concat = tf.reshape(cls_concat, [batch_size, 25, 25, 10, -1])  # [18, 25, 25, 10, 3]
    cls_predt_stack = tf.reshape(cls_concat, [batch_size, 25, 25, -1])  # [18, 25, 25, 30]

    bbox_concat = tf.concat((bbox_predt_conv3, bbox_predt_conv4, bbox_predt_conv5), 1)  # [18, 75, 25, 20]
    bbox_concat = tf.reshape(bbox_concat, [batch_size, 25, 25, 20, -1])  # [18, 25, 25, 20, 3]
    bbox_predt_stack = tf.reshape(bbox_concat, [batch_size, 25, 25, -1])  # [18, 25, 25, 60]
    # weight-fusion combines all outputs
    fused_cls_predt = Conv2D(filters=10, kernel_size=1, strides=1)(cls_predt_stack)  # [18, 25, 25, 10]
    fused_bbox_predt = Conv2D(filters=20, kernel_size=1, strides=1)(bbox_predt_stack)  # [18, 25, 25, 20]

    return fused_bbox_predt, fused_cls_predt