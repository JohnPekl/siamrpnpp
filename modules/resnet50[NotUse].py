
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Conv2D, MaxPooling2D, BatchNormalization, ReLU

def share_model(input, out_filters, stride = 1, down_sample = None, dilation = 1):
    # conv_1
    x = Conv2D(filters = out_filters, kernel_size = (1, 1), use_bias = False) (input)
    x = BatchNormalization() (x)
    x = ReLU() (x)
    # conv_2
    padding = 2 - stride
    if dilation > 1 and down_sample is not None:
        dilation = dilation // 2 # math.floor() or floor division
        padding = dilation
    if dilation > 1:
        padding = dilation
    pad_const = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
    x_pad = tf.pad(x, pad_const, "CONSTANT")
    x = Conv2D(filters = out_filters, kernel_size = (3, 3), strides = (stride, stride), use_bias = False, dilation_rate = dilation) (x_pad)
    x = BatchNormalization() (x)
    x = ReLU() (x)
    # conv_3
    x = Conv2D(filters = out_filters * 4, kernel_size = (1, 1), use_bias = False) (x)
    x = BatchNormalization() (x)
    # adding
    residual = input
    if down_sample is not None:
        residual = down_sample
    x = x + residual
    
    x = ReLU() (x)
    
    return x

def build_layers(input, out_filters, blocks, stride = 1, dilation = 1):
    expansion = 4  # fix
    in_filters = input.shape[-1] # It is initially with 64
    down_sample = None
    dilat_rate = dilation
    
    if (in_filters != out_filters * expansion) or (stride != 1):
        if (dilation == 1) and (stride == 1):
            down_sample = Conv2D(filters = out_filters * expansion, kernel_size = (1, 1), strides = (stride, stride), use_bias = False) (input)
            down_sample = BatchNormalization() (down_sample)
        else:
            dilat_rate = 1
            input_pad = input
            if dilation > 1:
                dilat_rate = dilation // 2
                padding = dilat_rate
                pad_const = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
                input_pad = tf.pad(input, pad_const, "CONSTANT")
            down_sample = Conv2D(filters = out_filters * expansion, kernel_size = (3, 3), strides = (stride, stride), use_bias = False, dilation_rate = dilat_rate) (input_pad)
            down_sample = BatchNormalization() (down_sample)
    
    x = share_model(input, out_filters, stride, down_sample, dilation)
    for idx in range(1, blocks):
        x = share_model(x, out_filters, dilation = dilation)
    
    return x

def resnet50(input, layers, used_layers):
    x = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), use_bias = False) (input)
    x_relu = ReLU() (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same') (x_relu)
    out_layer1 = build_layers(x, 64, layers[0])
    out_layer2 = build_layers(out_layer1, 128, layers[1], stride = 2)
    if 3 in used_layers:
        out_layer3 = build_layers(out_layer2, 256, layers[2], dilation = 2)
    if 4 in used_layers:
        out_layer4 = build_layers(out_layer3, 512, layers[3], dilation = 4)
    
    output = [x_relu, out_layer1, out_layer2, out_layer3, out_layer4]
    output = [output[idx] for idx in used_layers]
    if len(output) == 1:
        output = output[0]
    
    return output

def resnet50_model(input, layers, used_layers):
    x = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), use_bias = False) (input)
    x_relu = ReLU() (x)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same') (x_relu)
    out_layer1 = build_layers(x, 64, layers[0])
    out_layer2 = build_layers(out_layer1, 128, layers[1], stride = 2)
    if 3 in used_layers:
        out_layer3 = build_layers(out_layer2, 256, layers[2], dilation = 2)
    if 4 in used_layers:
        out_layer4 = build_layers(out_layer3, 512, layers[3], dilation = 4)
    
    output = [x_relu, out_layer1, out_layer2, out_layer3, out_layer4]
    output = [output[idx] for idx in used_layers]
    if len(output) == 1:
        output = output[0]
    
    model = Model(inputs = input, outputs = output)
    
    return model

# Use for testing purpose
import numpy as np #4D tensor with shape: (batch, channels, rows, cols)
image_test = np.empty(shape = [18, 537, 846, 3], dtype = np.float32)
image_test.fill(255)
with tf.device('/gpu:0'):
    output = resnet50(image_test, [3, 4, 6, 3], [2, 3, 4])
for idx in range(len(output)):
    print(output[idx])