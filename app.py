import streamlit as st
from io import BytesIO
from PIL import Image
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose, Add, Dropout
from tensorflow.keras import Model
import numpy as np
import cv2

st.title("Segmentasi Citra Optic")
#st.write("Hello world")

########################################################################################################
###############################################Model Arsitektur#########################################
def FCDUG(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    #encoder
    #block1
    con1 = Conv2D(32, (3, 3), activation='gelu', padding='same')(inputs)
    con1 = Conv2D(32, (3, 3), activation='gelu', padding='same')(con1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(con1)
    
    #block2
    con2 = Conv2D(64, (3, 3), activation='gelu', padding='same')(pool1)
    con2 = Conv2D(64, (3, 3), activation='gelu', padding='same')(con2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(con2)
    
    #block3
    con3 = Conv2D(128, (3, 3), activation='gelu', padding='same')(pool2)
    con3 = Conv2D(128, (3, 3), activation='gelu', padding='same')(con3)
    con3 = Conv2D(128, (3, 3), activation='gelu', padding='same')(con3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(con3)
    
    #block4
    con4 = Conv2D(256, (3, 3), activation='gelu', padding='same')(pool3)
    con4 = Conv2D(256, (3, 3), activation='gelu', padding='same')(con4)
    con4 = Conv2D(256, (3, 3), activation='gelu', padding='same')(con4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(con4)## (None, 14, 14, 512)
    
    #block5
    con5 = Conv2D(512, (3, 3), activation='gelu', padding='same')(pool4)
    con5 = Conv2D(512, (3, 3), activation='gelu', padding='same')(con5)
    con5 = Conv2D(512, (3, 3), activation='gelu', padding='same')(con5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(con5)## (None, 7, 7, 512)

    # D1
    cd1 = BatchNormalization()(pool5)
    cd1 = Activation("gelu")(cd1)
    cd1 = Conv2D(512, (1, 1), padding='same')(cd1)
    cd2 = BatchNormalization()(cd1)
    cd2 = Activation("gelu")(cd2)
    cd2 = Conv2D(512, (3, 3), padding='same')(cd2)
    merge_dense = concatenate([cd2,pool5], axis = 3)
   
    # D2
    cd3 = BatchNormalization()(merge_dense)
    cd3 = Activation("gelu")(cd3)
    cd3 = Conv2D(512, (1, 1), padding='same')(cd3)
    cd4 = BatchNormalization()(cd3)
    cd4 = Activation("gelu")(cd4)
    cd4 = Conv2D(512, (3, 3), padding='same')(cd4)
    merge_dense1 = concatenate([cd4,merge_dense], axis = 3)
    drop1 = Dropout(0.5)(merge_dense1)
   # D3
    cd5 = BatchNormalization()(drop1)
    cd5 = Activation("gelu")(cd5)
    cd5 = Conv2D(512, (1, 1), padding='same')(cd5)
    cd7 = BatchNormalization()(cd5)
    cd7 = Activation("gelu")(cd7)
    con6 = Conv2D(512, (3, 3), padding='same')(cd7)
    merge_dense2 = concatenate([cd7,merge_dense1], axis = 3)
    
    ## 4 times upsamping for pool4 layer
    con7_4 = Conv2DTranspose(512, kernel_size=(4,4),  strides=(4,4))(merge_dense2)
    
    ## 2 times upsampling for pool411
    pool411_2 = Conv2DTranspose(512 , kernel_size=(2,2),  strides=(2,2))(pool4)
    
    pool311 = Conv2D(512, (1 , 1) , activation='gelu' , padding='same', name="pool3_11")(pool3)
        
    o = Add(name="add")([pool411_2, pool311, con7_4 ])
    o = Conv2DTranspose(512, kernel_size=(8,8) ,  strides=(8,8))(o)
    
    
    o = Conv2D(3, 1, activation = 'softmax')(o)
    
    return Model(inputs=[inputs], outputs=[o])
    ########################################################################################################

model= FCDUG(input_size=(64,64,1))
model.summary()
model.load_weights("Model-fcdug.h5")

#input image

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    #st.write("nama file yang diupload=", uploaded_file.name)
    bytes_image = uploaded_file.read()

    image = Image.open(BytesIO(bytes_image))

    st.image(image, caption='Data Testing')

     # Preprocessing Image
    # Konversi kedalam array
    image_np  = np.asarray(image)
    # Grayscale
    image_np = np.mean(image_np, axis=-1, keepdims=True)


    # assuming your model expects input shape (None, 64, 64, 1)
    resized_image = cv2.resize(image_np, (64, 64))
    resized_image = np.expand_dims(resized_image, axis=-1)  # add a new dimension for grayscale channel
    resized_image = np.expand_dims(resized_image, axis=0)

    normalize_image = resized_image / 255.0



        # Proses
    output = model.predict(normalize_image)
    st.image(output)