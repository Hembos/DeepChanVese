from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Add, Input, Resizing, MaxPool2D, UpSampling2D
import tensorflow as tf
from keras import Model

def create_res_fpn_model(input, top_down_pyramid_size):
    c2, c3, c4, c5 = res_net_layers(input)

    p5 = Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c5p5')(c5)
    p4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(p5),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c4p4')(c4)])
    p3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(p4),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c3p3')(c3)])
    p2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(p3),
        Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c2p2')(c2)])

    p2 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p2")(p2)
    p3 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p3")(p3)
    p4 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p4")(p4)
    p5 = Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p5")(p5)

    p6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(p5)

    return p2, p3, p4, p5, p6

def res_net_layers(input):
    c1 = ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(input)
    c1 = Conv2D(64, 7, strides=2, use_bias=False, name="conv1_conv")(c1)
    c1 = BatchNormalization(name="conv1_bn")(c1, training=False)
    c1 = Activation("relu", name="conv1_relu")(c1)
    c1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(c1)
    c1 = MaxPooling2D(3, strides=2, name="pool1_pool", padding="same")(c1)

    c2 = create_shortcut_block(c1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    c2 = create_resnet_block(c2, 3, [64, 64, 256], stage=2, block='b')
    c2 = create_resnet_block(c2, 3, [64, 64, 256], stage=2, block='c')

    c3 = create_shortcut_block(c2, 3, [128, 128, 512], stage=3, block='a')
    c3 = create_resnet_block(c3, 3, [128, 128, 512], stage=3, block='b')
    c3 = create_resnet_block(c3, 3, [128, 128, 512], stage=3, block='c')
    c3 = create_resnet_block(c3, 3, [128, 128, 512], stage=3, block='d')

    c4 = create_shortcut_block(c3, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(5):
        c4 = create_resnet_block(c4, 3, [256, 256, 1024], stage=4, block=chr(98 + i))

    c5 = create_shortcut_block(c4, 3, [512, 512, 2048], stage=5, block='a')
    c5 = create_resnet_block(c5, 3, [512, 512, 2048], stage=5, block='b')
    c5 = create_resnet_block(c5, 3, [512, 512, 2048], stage=5, block='c')

    return [c2, c3, c4, c5]

def create_shortcut_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=False)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=False)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=False)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=False)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def create_resnet_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=False)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=False)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=False)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x