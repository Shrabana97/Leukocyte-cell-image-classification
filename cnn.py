import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2


def classical_cnn(lambd, dim, output_neurons, output_activation):
    print("\nTRAINING ON classical_cnn MODEL:-")
    
    def block(tensor, conv_reps, n_filters):
        x = Conv2D(filters = n_filters, kernel_regularizer=l2(lambd), bias_regularizer=l2(lambd), kernel_size = (3,3), padding = 'same')(tensor)
        x = LeakyReLU()(x)
        
        for i in range(conv_reps-1):
            x = Conv2D(filters = n_filters, kernel_regularizer=l2(lambd), bias_regularizer=l2(lambd), kernel_size = (3,3), padding = 'same')(x)
            x = LeakyReLU()(x)
            
        x = MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same')(x)
        x = Dropout(0.5)(x)
        return x

    input = Input(shape = dim)
        
    k = 8
    x = block(input, conv_reps = 1, n_filters = k)
    x = block(x, conv_reps = 1, n_filters = 2*k)
    x = block(x, conv_reps = 1, n_filters = 4*k)
    x = block(x, conv_reps = 1, n_filters = 8*k)
    x = block(x, conv_reps = 1, n_filters = 16*k)
    
    x = Flatten()(x)
    
    dense_reps = 2
    
    for i in range(dense_reps):
        x = Dense(128, kernel_regularizer=l2(lambd), bias_regularizer=l2(lambd))(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)
    
    output = Dense(output_neurons, output_activation)(x)  
    
    model = Model(inputs = input, outputs = output)
    
    return model
