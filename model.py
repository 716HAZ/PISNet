import tensorflow as tf
from loss import total_loss


class Stem(tf.keras.layers.Layer):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.re1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.re2 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        cv1 = self.conv1(inputs)
        bn1 = self.bn1(cv1)
        re1 = self.re1(bn1)
        cv2 = self.conv2(re1)
        bn2 = self.bn2(cv2)
        re2 = self.re2(bn2)
        return re2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'bn1': self.bn1,
            're1': self.re1,
            'conv2': self.conv2,
            'bn2': self.bn2,
            're2': self.re2
        })
        return config


class MFsm(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(MFsm, self).__init__()
        self.dense1 = tf.keras.layers.Dense(fil_num // 8, activation='relu', kernel_initializer='he_normal',
                                            use_bias=False, bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(fil_num, kernel_initializer='he_normal', use_bias=False,
                                            bias_initializer='zeros')
        self.dense3 = tf.keras.layers.Dense(fil_num // 8, activation='relu', kernel_initializer='he_normal',
                                            use_bias=False, bias_initializer='zeros')
        self.dense4 = tf.keras.layers.Dense(fil_num, kernel_initializer='he_normal', use_bias=False,
                                            bias_initializer='zeros')
        self.conv = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        shape = inputs.shape
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        max_pool = tf.keras.layers.MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
        max_pool = self.dense3(max_pool)
        max_pool = self.dense4(max_pool)
        add = tf.keras.layers.Add()([avg_pool, max_pool])
        sg = tf.keras.layers.Activation('sigmoid')(add)
        out = sg * inputs
        out = out + inputs
        out = self.conv(out)
        out = self.bn(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense1': self.dense1,
            'dense2': self.dense2,
            'dense3': self.dense3,
            'dense4': self.dense4,
            'conv': self.conv,
            'bn': self.bn
        })
        return config


class CBA1(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(CBA1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=3, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.re1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.re2 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        cv1 = self.conv1(inputs)
        bn1 = self.bn1(cv1)
        re1 = self.re1(bn1)
        cv2 = self.conv2(re1)
        bn2 = self.bn2(cv2)
        re2 = self.re2(bn2)
        return re2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'bn1': self.bn1,
            're1': self.re1,
            'conv2': self.conv2,
            'bn2': self.bn2,
            're2': self.re2
        })
        return config


class CBA2(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(CBA2, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.re = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        cv = self.conv(inputs)
        bn = self.bn(cv)
        re = self.re(bn)
        return re

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv': self.conv,
            'bn': self.bn,
            're': self.re,
            })
        return config


class CBA3(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(CBA3, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=1, padding='same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.re = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        cv = self.conv(inputs)
        bn = self.bn(cv)
        re = self.re(bn)
        return re

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv': self.conv,
            'bn': self.bn,
            're': self.re,
            })
        return config


class MHSA(tf.keras.layers.Layer):
    def __init__(self, dim, head_dim, attn_drop):
        super(MHSA, self).__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)

    def call(self, inputs):
        _, N, C = inputs.shape[0], inputs.shape[1], inputs.shape[2]

        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        out = tf.transpose(x, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, N, C])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'scale': self.scale,
            'qkb': self.qkv,
            'attn_drop': self.attn_drop,
        })
        return config


class TB(tf.keras.layers.Layer):
    def __init__(self, fil_num, head_dim, attn_drop):
        super(TB, self).__init__()
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.mhsa = MHSA(fil_num, head_dim, attn_drop)
        self.proj = tf.keras.layers.Dense(fil_num)

    def call(self, inputs):
        _, H, W, C = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
        x = self.BN1(inputs)
        x = tf.reshape(x, [-1, H*W, C])
        x = self.mhsa(x)
        x = self.proj(x)
        out = tf.reshape(x, [-1, H, W, C])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'BN1': self.BN1,
            'proj': self.proj,
            'mhsa': self.mhsa,
        })
        return config


class SPN(tf.keras.layers.Layer):
    def __init__(self):
        super(SPN, self).__init__()
        self.stem = Stem()

        self.sp1 = MFsm(256)
        self.sp2 = MFsm(128)
        self.sp3 = CBA2(256)
        self.sp4 = CBA2(512)
        # self.sp3 = TB(256, 16, 0.02)
        # self.sp4 = TB(512, 32, 0.02)
        self.sp5 = MFsm(256)
        self.sp6 = MFsm(128)

        self.cv1 = CBA1(128)
        self.cv2 = CBA1(256)
        self.cv3 = CBA1(512)
        self.cv4 = CBA2(256)
        self.cv5 = CBA1(128)
        self.cv6 = CBA2(256)
        self.cv7 = CBA1(512)
        self.cv8 = CBA1(256)
        self.cv9 = CBA1(128)

        self.out1 = CBA3(256)
        self.out2 = CBA3(256)
        self.out3 = CBA3(256)

    def call(self, inputs):
        st = self.stem(inputs)

        mp1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(st)
        cv1 = self.cv1(mp1)
        mp2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cv1)
        cv2 = self.cv2(mp2)
        mp3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cv2)
        cv3 = self.cv3(mp3)

        up1 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(cv3)
        skip1 = self.sp1(cv2)
        concat1 = tf.keras.layers.Concatenate()([skip1, up1])
        cv4 = self.cv4(concat1)
        up2 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(cv4)
        skip2 = self.sp2(cv1)
        concat2 = tf.keras.layers.Concatenate()([skip2, up2])
        cv5 = self.cv5(concat2)

        mp4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cv5)
        skip3 = self.sp3(cv4)
        concat3 = tf.keras.layers.Concatenate()([skip3, mp4])
        cv6 = self.cv6(concat3)
        mp5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cv6)
        skip4 = self.sp4(cv3)
        concat4 = tf.keras.layers.Concatenate()([skip4, mp5])
        cv7 = self.cv7(concat4)

        up3 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(cv7)
        skip5 = self.sp5(cv6)
        concat5 = tf.keras.layers.Concatenate()([skip5, up3])
        cv8 = self.cv8(concat5)
        up4 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(cv8)
        skip6 = self.sp6(cv5)
        concat6 = tf.keras.layers.Concatenate()([skip6, up4])
        cv9 = self.cv9(concat6)

        up5 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(cv9)
        out1 = self.out1(cv7)
        out2 = self.out2(cv8)
        out3 = self.out3(cv9)

        return out1, out2, out3, up5

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'stem': self.stem,
            'sp1': self.sp1,
            'sp2': self.sp2,
            'sp3': self.sp3,
            'sp4': self.sp4,
            'sp5': self.sp5,
            'sp6': self.sp6,
            'cv1': self.cv1,
            'cv2': self.cv2,
            'cv3': self.cv3,
            'cv4': self.cv4,
            'cv5': self.cv5,
            'cv6': self.cv6,
            'cv7': self.cv7,
            'cv8': self.cv8,
            'cv9': self.cv9,
            'out1': self.out1,
            'out2': self.out2,
            'out3': self.out3,
            })
        return config


def yolo_body(input_shape, anchors_mask, num_classes):
    inputs = tf.keras.layers.Input(input_shape)

    out1, out2, out3, out4 = SPN().call(inputs)

    out1 = tf.keras.layers.Conv2D(filters=len(anchors_mask[2]) * (5 + num_classes),
                                  kernel_size=1, strides=1, padding="same")(out1)
    out2 = tf.keras.layers.Conv2D(filters=len(anchors_mask[1]) * (5 + num_classes),
                                  kernel_size=1, strides=1, padding="same")(out2)
    out3 = tf.keras.layers.Conv2D(filters=len(anchors_mask[0]) * (5 + num_classes),
                                  kernel_size=1, strides=1, padding="same")(out3)

    out4 = tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2),
                                                                padding='same', use_bias=False),
                                tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu"),
                                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2),
                                                                padding='same', use_bias=False),
                                tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu")])(out4)
    out4 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False),
                                tf.keras.layers.Activation("sigmoid")])(out4)

    return tf.keras.models.Model(inputs, [out1, out2, out3, out4])


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    box_true = [tf.keras.layers.Input(shape=(input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    mask_true = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    y_true = [box_true, mask_true]

    model_loss = tf.keras.layers.Lambda(total_loss, output_shape=(1, ), name='yolo_loss',
                                        arguments={'input_shape': input_shape, 'anchors': anchors,
                                                   'anchors_mask': anchors_mask, 'num_classes': num_classes,
                                                   'box_ratio': 0.05, 'obj_ratio': 1, 'cls_ratio': 0.5})\
        ([*model_body.output, *y_true])
    model = tf.keras.models.Model([model_body.input, *y_true], model_loss)
    return model


if __name__ == "__main__":
    input_shape = [256, 512, 3]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 1
    model = yolo_body(input_shape, anchors_mask, num_classes)
    model.summary()
