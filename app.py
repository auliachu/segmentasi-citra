import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from tensorflow.keras.layers import Activation, Dropout, Conv2DTranspose, Add
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model

# Konfigurasi Halaman Web
# Untuk mengatur Judul Web, Icon web, jenis Layout, etc
st.set_page_config(
    page_title="Segmentasi Citra",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://google.com/',
        'Report a bug': "mailto:riyadfebrian@gmail.com",
        'About': "Simple ML Apps"
    }
)

def FCDUG(input_size=(256,256,1)):
    """
    Arsitektur Model FCDUG
    """

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

@st.cache_resource
def load_models():
    """
    @st.cache_resource decorator digunakan untuk menyimpan resource model.

    Fungsi load_models() akan membuat model FCDUG dan menerapkan weights dari file .h5 

    """
    model = FCDUG(input_size=(64,64,1))
    model.load_weights("Model-fcdug.h5")

    return model

# global variable model yang bisa diakses oleh fungsi/method
model = load_models()

def preprocess_image(image_predict):
    """
    Fungsi preprocess_image adalah untuk mempersiapkan input sebelum dimasukkan kedalam model.

    """

    # Grayscaling
    image_np = np.mean(image_predict, axis=-1, keepdims=True)
    # Resize ukuran gambar ke 64 x 64 
    resized_image = cv2.resize(image_np, (64, 64))
    # Normalisasi Gambar
    normalized_image = resized_image.astype('float32') / 255.0
    # Menambah dimensi input untuk menyesuaikan inputan deep learning
    input_image = np.expand_dims(normalized_image, axis=-1)  
    input_image = np.expand_dims(input_image, axis=0)

    return input_image

@st.cache_data()
def predict(image_predict):
    """
    @st.cache_data decorator berfungsi untuk caching / menyimpan data prediksi sementara

    Fungsi predict digunakan untuk melakukan prediksi data
    """
    image = preprocess_image(image_predict)
    prediction = model.predict(image)
    
    return prediction
    

def main():
    """
    Kode Utama yang menampilkan UI halaman web
    seperti Judul teks, Input untuk Memilih Gambar, dan Output Gambar
    """
    # Menampilkan Judul teks pada Konten halaman web
    st.title('Segmentasi Citra Optic Disk')

    # Menampilkan Form untuk Mengupload File
    uploaded_file = st.file_uploader("Choose image file")

    # Seleksi kondisi untuk mengecek apakah user telah mengupload sebuah file
    if uploaded_file is not None:
        # Pengecekan apakah file yang di upload berformat jpg atau png 
        if uploaded_file.name.endswith('jpg') or uploaded_file.name.endswith('png'):
            
            # Membaca file kedalam bentuk Bytes
            file_contents = uploaded_file.read()
            # Membuka objek gambar dari bytes
            image_predict = Image.open(BytesIO(file_contents))
            # Konversi gambar kedalam bentuk array
            image_predict  = np.asarray(image_predict)

            # Prediksi
            output = predict(image_predict)

            # Tampilkan Input Gambar
            st.write("Input")
            st.image(image_predict)

            # Tampilkan Prediksi Segmentasi Gambar
            st.write("Prediction")
            st.image(output)


        else:
            # Jika file yang di upload bukan gambar, tampilkan pesan warning berikut
            st.warning("File is not an image")


if __name__ == '__main__':
    # Kode yang berada dibawah ini akan dijalankan pertama kali ketika app.py dijalankan
    
    main()


