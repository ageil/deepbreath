from keras.models import Model
from keras.layers import Input, Activation, Add, AveragePooling3D
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv3D, Cropping3D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.pooling import MaxPooling3D
from keras.initializers import glorot_normal


def gapnet(downsample=1):
    assert downsample > 0, "input downsampling factor must be 1 (none) or larger"

    # block 1
    input_1 = Input(shape=(1, 142, 322, 262), name="input_1")  # timesteps x channels x imgsize
    mpool = MaxPooling3D(pool_size=(downsample, downsample, downsample), data_format="channels_first", name="mpool3D")(input_1)
    conv_1 = Conv3D(filters=6, kernel_size=(1, 5, 5), data_format="channels_first", name="conv_1")(mpool)
    relu_2 = Activation('relu', name="ReLU_2")(conv_1)
    down_3 = Conv3D(filters=8, kernel_size=(1, 1, 1), strides=(2, 2, 2), data_format="channels_first", name="down_3")(relu_2)

    # block 2
    relu_4 = Activation('relu', name="ReLU_4")(down_3)
    conv_5 = Conv3D(filters=8, kernel_size=(1, 5, 5), data_format="channels_first", name="conv_5")(relu_4)
    relu_6 = Activation("relu", name="ReLU_6")(conv_5)
    conv_7 = Conv3D(filters=8, kernel_size=(1, 5, 5), data_format="channels_first", name="conv_7")(relu_6)

    # skip connection 1
    crop_9 = Cropping3D(cropping=((0, 0), (4, 4), (4, 4)), data_format="channels_first", name="crop_9")(down_3)
    merge_8 = Add(name="merge_8")([conv_7, crop_9])

    # block 3
    relu_10 = Activation("relu", name="ReLU_10")(merge_8)
    conv_11 = Conv3D(filters=8, kernel_size=(1, 5, 5), data_format="channels_first", name="conv_11")(relu_10)
    relu_12 = Activation("relu", name="ReLU_12")(conv_11)
    conv_13 = Conv3D(filters=8, kernel_size=(1, 5, 5), data_format="channels_first", name="conv_13")(relu_12)

    # skip connection 2
    crop_15 = Cropping3D(cropping=((0, 0), (4, 4), (4, 4)), data_format="channels_first", name="crop_15")(merge_8)
    merge_14 = Add(name="merge_14")([conv_13, crop_15])

    # block 4
    relu_16 = Activation("relu", name="ReLU_16")(merge_14)
    down_17 = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(1, 2, 2), data_format="channels_first", name="down_17")(relu_16)
    relu_18 = Activation("relu", name="ReLU_18")(down_17)
    conv_19 = Conv3D(filters=16, kernel_size=(1, 3, 3), data_format="channels_first", name="conv_19")(relu_18)
    relu_20 = Activation("relu", name="ReLU_20")(conv_19)
    conv_21 = Conv3D(filters=16, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_21")(relu_20)

    # skip connection 3
    crop_23 = Cropping3D(cropping=((0, 0), (1, 1), (1, 1)), data_format="channels_first", name="crop_23")(down_17)
    merge_22 = Add(name="merge_22")([conv_21, crop_23])

    # block 5
    relu_24 = Activation("relu", name="ReLU_24")(merge_22)
    conv_25 = Conv3D(filters=16, kernel_size=(1, 3, 3), data_format="channels_first", name="conv_25")(relu_24)
    relu_26 = Activation("relu", name="ReLU_26")(conv_25)
    conv_27 = Conv3D(filters=16, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_27")(relu_26)

    # skip connection 4
    crop_29 = Cropping3D(cropping=((0, 0), (1, 1), (1, 1)), data_format="channels_first", name="crop_29")(merge_22)
    merge_28 = Add(name="merge_28")([conv_27, crop_29])

    # block 6
    relu_30 = Activation("relu", name="ReLU_30")(merge_28)
    down_31 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=(1, 2, 2), data_format="channels_first", name="down_31")(relu_30)
    relu_32 = Activation("relu", name="ReLU_32")(down_31)
    conv_33 = Conv3D(filters=32, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_33")(relu_32)
    relu_34 = Activation("relu", name="ReLU_34")(conv_33)
    conv_35 = Conv3D(filters=32, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_35")(relu_34)

    # skip connection 5
    merge_36 = Add(name="merge_36")([conv_35, down_31])

    # block 7
    relu_37 = Activation("relu", name="ReLU_37")(merge_36)
    conv_38 = Conv3D(filters=32, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_38")(relu_37)
    relu_39 = Activation("relu", name="ReLU_39")(conv_38)
    conv_40 = Conv3D(filters=32, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_40")(relu_39)

    # skip connection 6
    merge_41 = Add(name="merge_41")([conv_40, merge_36])

    # block 8
    relu_42 = Activation("relu", name="ReLU_42")(merge_41)
    print("relu_42:", relu_42._keras_shape)
    down_43 = AveragePooling3D(pool_size=relu_42._keras_shape[-3:], data_format="channels_first", name="down_43")(relu_42)
    print("down_43:", down_43._keras_shape)
    output_44 = Conv3D(filters=1, kernel_size=(1, 1, 1), data_format="channels_first", name="conv_44")(down_43)
    print("output_44:", output_44._keras_shape)

    model = Model(inputs=input_1, outputs=output_44)

    return model