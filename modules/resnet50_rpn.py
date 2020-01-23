
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Conv2D, MaxPooling2D, BatchNormalization, ReLU


def Batch_Norm(x, training):
    x = tf.layers.batch_normalization(x, training = training)
    return x

def Bottleneck(input, out_filters, stride = 1, down_sample = None, padding = 1, dilation = 1, training = True):
    # conv_1
    x = Conv2D(filters = out_filters, kernel_size = (1, 1), use_bias = False) (input)
    x = Batch_Norm(x, training)
    x = ReLU() (x)
    # conv_2
    x_pad = x
    if padding != 0:
        pad_const = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
        x_pad = tf.pad(x, pad_const, "CONSTANT")
    x = Conv2D(filters = out_filters, kernel_size = (3, 3), strides = (stride, stride), use_bias = False, dilation_rate = dilation) (x_pad)
    x = Batch_Norm(x, training) #BatchNormalization() (x)
    x = ReLU() (x)
    # conv_3
    x = Conv2D(filters = out_filters * 4, kernel_size = (1, 1), use_bias = False) (x)
    x = Batch_Norm(x, training) #BatchNormalization() (x)
    # adding
    residual = input
    if down_sample is not None:
        residual = down_sample
    x = x + residual
    
    x = ReLU() (x)
    
    return x

def build_layers(input, out_filters, blocks, ksize = 1, stride = 1, padding = 1, dilation = 1, training = True):
    expansion = 4  # fix
    in_filters = input.shape[-1] # It is initially with 64
    down_sample = None
    
    if (in_filters != out_filters * expansion) or (stride != 1):
        down_sample = Conv2D(filters = out_filters * expansion, kernel_size = (ksize, ksize), strides = (stride, stride), use_bias = False, dilation_rate = dilation) (input)
        down_sample = Batch_Norm(down_sample, training) #BatchNormalization() (down_sample)
    
    x = Bottleneck(input, out_filters, stride, down_sample, padding, dilation, training = training)
    for idx in range(1, blocks):
        x = Bottleneck(x, out_filters, training = training)
    
    return x

def resnet50(input, training = True):
    layers = [3, 4, 6, 3]
    x = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), use_bias = False) (input)
    x = Batch_Norm(x, training) #BatchNormalization() (x)
    x = ReLU() (x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same') (x)
    
    out_layer1 = build_layers(x, 64, layers[0], training = training)
    
    out_layer2 = build_layers(out_layer1, 128, layers[1], ksize = 3, padding = 0, stride = 2, training = training)
    conv_3 = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), use_bias = False) (out_layer2)
    out_layer3 = build_layers(out_layer2, 256, layers[2], padding = 2, dilation = 2, training = training)
    conv_4 = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), use_bias = False) (out_layer3)
    out_layer4 = build_layers(out_layer3, 512, layers[3], padding = 4, dilation = 4, training = training)
    conv_5 = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), use_bias = False) (out_layer4)
    
    return conv_3, conv_4, conv_5
    
    #output = [conv_3, conv_4, conv_5]
    #model = Model(inputs = input, outputs = output)
    #return model

# Input (batch, height, width, channels) by default "channels_last"
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D