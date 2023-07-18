import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('mymodel2.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # AI Image Classification
         """
         )

with open('style.css') as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

file = st.file_uploader('Please upload an image', type=["jpg", "png", "jpeg", "webm"],)
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size)
        img = np.asarray(image)
        img=img/255
        img=np.expand_dims(img,[0])
    
        prediction = model.predict(img)
        
        return prediction
if file is None:
    st.text('Please upload an image file')
else:
    image = Image.open(file)
    image = image.convert("RGB")
    st.image(image, use_column_width=True)
    try:
      predictions = import_and_predict(image, model)
      score = tf.nn.softmax(predictions[0])
      predictions = np.argmax(predictions, axis = 1)
      if(predictions == 0):
          st.write('<p class = "prediction">The image is most likely an AI Generated Image</p>', unsafe_allow_html=True)
      else:
          st.write('<p class = "prediction">The image is most likely a Real Image</p>', unsafe_allow_html=True)
    except Exception as e:
        st.write(f'An error occurred during prediction')