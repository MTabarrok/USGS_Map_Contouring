import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title('HSV Color Masking and Contour Detection')

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "tif"])
fileinfo = uploaded_file.name[:-8]

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img_array, caption=f'{uploaded_file.name}', use_column_width=True)

    st.subheader('HSV Masking')

    h_min = st.number_input('Hue min', min_value=0, max_value=180, value=0)
    h_max = st.number_input('Hue max', min_value=0, max_value=180, value=180)
    s_min = st.number_input('Saturation min', min_value=0, max_value=255, value=0)
    s_max = st.number_input('Saturation max', min_value=0, max_value=255, value=255)
    v_min = st.number_input('Value min', min_value=0, max_value=255, value=0)
    v_max = st.number_input('Value max', min_value=0, max_value=255, value=255)

    # Convert image to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # Define color range for masking
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])

    # Apply mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
    masked_image = cv2.bitwise_and(img_array, img_array, mask=mask)

    st.subheader('Masked Image')
    st.image(masked_image, caption='Masked Image', use_column_width=True)

        # Option to save the masked image
    if st.button('Save Masked Image'):
        masked_img = Image.fromarray(mask)
        masked_img.save(f'C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclemation/NewYorkMaps/Masks/{fileinfo}-HSV[{lower_range}][{upper_range}].tif')
        st.success('Image saved as masked_image.tif')

    st.subheader('Morphological Operations')

    kernel_size = st.slider('Kernel size for morphological operations', min_value=0, max_value=20, value=5)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    # Perform morphological opening to remove small artifacts
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    st.subheader('Image after Morphological Operations')
    st.image(morph, caption='Image after Morphological Operations', use_column_width=True)

    st.subheader('Contour Detection')

    min_contour_area = st.number_input('Minimum contour area', min_value=1, max_value=5000, value=1000)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw all contours on a copy of the original image
    img_with_contours = np.copy(img_array)
    cv2.drawContours(img_with_contours, contours, -1, (0,255,0), 3)

    # Assuming 'contours' is your list of contours and 'img' is your original image

    if st.button('Save Countour Image'):
        masked_img = Image.fromarray(img_with_contours)
        masked_img.save(f'C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclemation/NewYorkMaps/Contours/{fileinfo}-Contour[{min_contour_area}]-HSV[{lower_range}][{upper_range}]masked_image.tif')
        st.success('Image saved as masked_image.tif')

    st.subheader('Image with Contours')
    st.image(img_with_contours, caption='Image with Contours', use_column_width=True)

    
    width = img.width
    height = img.height

    st.subheader(f'Bounding Box {width} x {height}')

    tlx = st.number_input('tlx', min_value=0, max_value=width, value=380)
    tly = st.number_input('tly', min_value=0, max_value=height, value=260)
    brx = st.number_input('brx', min_value=0, max_value=width, value=4375)
    bry = st.number_input('bry', min_value=0, max_value=height, value=5495)

    # Create a black image with the same dimensions as the original image
    img_black = np.zeros_like(img_array)
    # Draw the contours on the black image
    img_contours = cv2.drawContours(img_black, contours, -1, (0, 255, 0), 3)
    img_contours = img_contours[tly:bry, tlx:brx]
    #img_contours = cv2.rectangle(img_contours, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    st.subheader('Image with Bounding Box')
    st.image(img_contours, caption='Image with Bounding Box', use_column_width=True)



    if st.button('Save Countour Image Black Background and Bounding Box'):
        masked_img = Image.fromarray(img_contours)
        masked_img.save(f'C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclamation/NewYorkMaps/Areas/{fileinfo}-bounds[{tlx},{tly}]x[{brx},{bry}]-HSV{lower_range}{upper_range}-kernel[{kernel_size}]-contour[{min_contour_area}].tif')
        st.success(f'Image saved as C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclamation/NewYorkMaps/Areas/{fileinfo}-bounds[{tlx},{tly}]x[{brx},{bry}]-HSV{lower_range}{upper_range}-kernel[{kernel_size}]-contour[{min_contour_area}].tif')

    AreaFilled = st.file_uploader("Choose a file", type=["jpg", "png", "tif"])

    if AreaFilled is not None:
        Areaimg = Image.open(AreaFilled)
        Areaimg_array = np.array(Areaimg)
        st.image(Areaimg_array, caption='Filled Areas', use_column_width=True)
    
        st.subheader('RGB Masking')
    
        r_min = st.slider('R min', min_value=0, max_value=255, value=0)  # Changed max_value to 255
        r_max = st.slider('R max', min_value=0, max_value=255, value=255)  # Changed max_value to 255
        g_min = st.slider('G min', min_value=0, max_value=255, value=0)
        g_max = st.slider('G max', min_value=0, max_value=255, value=255)
        b_min = st.slider('B min', min_value=0, max_value=255, value=0)
        b_max = st.slider('B max', min_value=0, max_value=255, value=255)
    
        # Define color range for masking
        lower_range = np.array([b_min, g_min, r_min])
        upper_range = np.array([b_max, g_max, r_max])
    
        # Apply mask
        bgr = cv2.cvtColor(Areaimg_array, cv2.COLOR_RGB2BGR)
        RGBmask = cv2.inRange(bgr, lower_range, upper_range)
        bgr_masked_image = cv2.bitwise_and(bgr, bgr, mask=RGBmask)
    
        # Convert back to RGB for displaying
        RGBmasked_image = cv2.cvtColor(bgr_masked_image, cv2.COLOR_BGR2RGB)
    
        st.subheader('Masked Image')
        st.image(RGBmasked_image, caption='Masked Image', use_column_width=True)

    
        # Calculate the total and red pixel count within the image
        total_pixels = Areaimg.width * Areaimg.height
        red_pixels = np.sum(RGBmasked_image[:,:,0] > r_min)  # Sum over the red channel
    
        # Calculate the land percentage
        land_percentage = (red_pixels / total_pixels) * 100
        land_percentage = str(land_percentage)[:5]
        st.subheader(f'Land percentage = {land_percentage}')
        
        if st.button('Save Red Isolated Image'):
            masked_img = Image.fromarray(RGBmasked_image)
            masked_img.save(f'C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclamation/NewYorkMaps/OnlyRed/{fileinfo}-landpercent[{land_percentage}]-bounds[{tlx},{tly}]x[{brx},{bry}]-HSV{lower_range}{upper_range}-kernel[{kernel_size}]-contour[{min_contour_area}].tif')
            st.success(f'Image saved as C:/Users/maxwe/AppData/Local/Programs/Python/Python311/Scripts/Projects/LandReclamation/NewYorkMaps/OnlyRed/{fileinfo}-landpercent[{land_percentage}]-bounds[{tlx},{tly}]x[{brx},{bry}]-HSV{lower_range}{upper_range}-kernel[{kernel_size}]-contour[{min_contour_area}].tif')



