from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Add, Input
import tensorflow as tf
from keras import Model

class ResFPN(Model):
    def __init__(self, input_shape):
        super(ResFPN, self).__init__()
        self.activation = 'relu'
        self.in_shape = input_shape

        self.model_layers = {}

        self.model_layers["conv1_pad"] = ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")
        self.model_layers["conv1_conv"] = Conv2D(64, 7, strides=2, use_bias=False, name="conv1_conv")

        self.model_layers["conv1_bn"] = BatchNormalization(axis=3, epsilon=1.001e-5, name="conv1_bn")
        self.model_layers["conv1_" + self.activation] = Activation(self.activation, name="conv1_" + self.activation)

        self.model_layers["pool1_pad"] = ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")
        self.model_layers["pool1_pool"] = MaxPooling2D(3, strides=2, name="pool1_pool")

        self.create_stack(64, 3, stride1=1, name="conv2")
        self.create_stack(128, 4, name="conv3")
        self.create_stack(256, 6, name="conv4")
        self.create_stack(512, 3, name="conv5")

        # self.model_layers["conv2_lateral"] = Conv2D()

    # def create_fpn_block(self):
        

    def create_res_block(self, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        if conv_shortcut:
            self.model_layers[name + "_0_conv"] = Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")
            self.model_layers[name + "_0_bn"] = BatchNormalization(axis=3, epsilon=1.001e-5, name=name + "_0_bn")

        self.model_layers[name + "_1_conv"] = Conv2D(filters, 1, strides=stride, name=name + "_1_conv")
        self.model_layers[name + "_1_bn"] = BatchNormalization(axis=3, epsilon=1.001e-5, name=name + "_1_bn")
        self.model_layers[name + "_1_" + self.activation] = Activation(self.activation, name=name + "_1_" + self.activation)

        self.model_layers[name + "_2_conv"] = Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")
        self.model_layers[name + "_2_bn"] = BatchNormalization(axis=3, epsilon=1.001e-5, name=name + "_2_bn")
        self.model_layers[name + "_2_" + self.activation] = Activation(self.activation, name=name + "_2_" + self.activation)

        self.model_layers[name + "_3_conv"]  = Conv2D(4 * filters, 1, name=name + "_3_conv")
        self.model_layers[name + "_3_bn"] = BatchNormalization(axis=3, epsilon=1.001e-5, name=name + "_3_bn")

        self.model_layers[name + "_add"] = Add(name=name + "_add")
        self.model_layers[name + "_out"] = Activation(self.activation, name=name + "_out")

    def create_stack(self, filters, blocks, stride1=2, name=None):
        self.create_res_block(filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            self.create_res_block(filters, conv_shortcut=False, name=name + "_block" + str(i))

    def summary(self):
        x = Input(shape=self.in_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def call_resnet_stack(self, input, stack_ind, training, block_num):
        stack_name = f'conv{stack_ind}'
        shortcut = self.model_layers[stack_name + "_block1_0_conv"](input)
        shortcut = self.model_layers[stack_name + "_block1_0_bn"](shortcut, training=training)

        x = input
        for j in range(1, block_num + 1):
            if j != 1:
                shortcut = x

            x = self.model_layers[stack_name + f"_block{j}" + "_1_conv"](x)
            x = self.model_layers[stack_name + f"_block{j}" + "_1_bn"](x, training=training)
            x = self.model_layers[stack_name + f"_block{j}" + "_1_relu"](x)

            x = self.model_layers[stack_name + f"_block{j}" + "_2_conv"](x)
            x = self.model_layers[stack_name + f"_block{j}" + "_2_bn"](x, training=training)
            x = self.model_layers[stack_name + f"_block{j}" + "_2_relu"](x)

            x = self.model_layers[stack_name + f"_block{j}" + "_3_conv"](x)
            x = self.model_layers[stack_name + f"_block{j}" + "_3_bn"](x, training=training)

            x = self.model_layers[stack_name + f"_block{j}" + "_add"]([shortcut, x])
            x = self.model_layers[stack_name + f"_block{j}" + "_out"](x)

        return x


    def call(self, inputs, training=False):
        x = self.model_layers["conv1_pad"](inputs)
        x = self.model_layers["conv1_conv"](x)
        x = self.model_layers["conv1_bn"](x, training=training)
        x = self.model_layers["conv1_relu"](x)
        x = self.model_layers["pool1_pad"](x)
        c1 = self.model_layers["pool1_pool"](x)
        c2 = self.call_resnet_stack(c1, 2, training, 3)
        c3 = self.call_resnet_stack(c2, 3, training, 4)
        c4 = self.call_resnet_stack(c3, 4, training, 6)
        c5 = self.call_resnet_stack(c4, 5, training, 3)

        
            
        return c5