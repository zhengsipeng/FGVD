from keras.layers import Conv2D, MaxPool2D
from keras import backend as K

def nn_base(input_tensor=None, trainable=False):
    if K.image_dim_ordering == 'th':
        input_shape = [3, None, None]
    else:
        input_shape = [None, None, 3]
    
    
    conv_a_1 = Conv2D(64, (7, 7), name='conv_a_1')(pad(input_a, 3))
    conv_a_2 = Conv2D(128, (5, 5), name='conv_a_2')(pad(conv_a_1, 2))
    conv_a_3 = Conv2D(256, (5, 5), name='conv_a_2')(pad(conv_a_2, 2))
    
    conv_b_1 = Conv2D(64, (7, 7), name='conv_b_1')(pad(input_b, 3))
    conv_b_2 = Conv2D(128, (5, 5), name='conv_b_2')(pad(input_b, 2))
    conv_b_3 = Conv2D(256, (5, 5), name='conv_b_3')(pad(input_b, 2))
    
    # Compute cross correlation with leaky relu activation
    cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
    cc_relu = LeakyReLU(cc)
    #Start Refinement Network