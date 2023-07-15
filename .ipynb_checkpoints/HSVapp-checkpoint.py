import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Load the image
uploaded_file = st.file_uploader("Choose an image...", type="tif")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Define range for color in HSV
    lower_hue = st.sidebar.slider('Lower Hue', 0, 180, 40)
    upper_hue = st.sidebar.slider('Upper Hue', 0, 180, 180)
    lower_saturation = st.sidebar.slider('Lower Saturation', 0, 255, 50)
    upper_saturation = st.sidebar.slider('Upper Saturation', 0, 255, 255)
    lower_value = st.sidebar.slider('Lower Value', 0, 255, 50)
    upper_value = st.sidebar.slider('Upper Value', 0, 255, 255)
    lower_color = np.array([lower_hue, lower_saturation, lower_value])
    upper_color = np.array([upper_hue, upper_saturation, upper_value])

    # Create a mask
    mask_color = cv2.inRange(image_hsv, lower_color, upper_color)

    # Apply the mask to get the colored areas
    colored_areas = cv2.bitwise_and(image_array, image_array, mask=mask_color)

    # Display the original image and the masked image
    st.image(image_array, use_column_width=True, caption='Original Image')
    st.image(colored_areas, use_column_width=True, caption='Masked Image')

    # Option to save the masked image
    if st.button('Save Masked Image'):
        masked_img = Image.fromarray(colored_areas)
        masked_img.save(f'LandReclamation/{lower_color}{upper_color}masked_image.tif')
        st.success('Image saved as masked_image.tif')
