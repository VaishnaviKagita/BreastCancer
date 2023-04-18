import tempfile
from tracemalloc import start
import streamlit as st
import cv2

import tensorflow
from tensorflow import keras
import numpy as np



from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

def run(source = None):

    model = tensorflow.keras.models.load_model(r'C:\Users\vaish\modelcnn.hdf5')

    img = cv2.imread(source)
    #img = io.imread(source)
    
    print(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img_gray, (5, 5), 0)   
    thresh = cv2.threshold(gray, 45, 25, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    img_half = cv2.resize(img, (100, 100))
    img_half = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY)
    x = img_to_array(img_half)
    
    x = np.array(x).reshape(-1, 100, 100, 1)

    
    y_pred = model.predict(x)
    y_pred
    if y_pred > 0.5:
        y_pred = 1
        print("D")
        
        return "Defect"
    else:
        y_pred = 0
        print("N")
        
        return "Normal"
        

    
#run(source=r"C:\Users\vaish\Downloads\data\Normal\10254_idx5_x101_y1151_class0.png")

# if __name__ == "__main__":
#     # opt = parse_opt()
#     # main(opt)
#     run(source=r'A:/TechieYan projects/AI/Completed/Identifying Defects in the Various Fabrics using Convolutional Neural Networks/FABRIC - Copy/Defect/aug_0_55.png')


def gui():
		
	st.title('BREAST CANCER DETECTION')
	st.sidebar.title('Parameters for Detection')
	st.sidebar.markdown('---')
	st.sidebar.title("Image Source")

	name = st.text_input("Enter your name:")
	contact = st.text_input("Enter your contact information:")

	with st.sidebar:
		st.markdown(
			'''
			<style>
			    [data-testid='sidebar'][aria-expanded='true'] > div:first-child {
			        width: 400px;
			         /* Baby pink color */
			    }
			    
			    [data-testid='sidebar'][aria-expanded='false'] > div:first-child {
			        width: 400px;
			        margin-left: -400px;
			        
			    }
			</style>
			''',
			unsafe_allow_html=True
		)

	source = st.sidebar.file_uploader('Upload Image', type=['jpg','png','jpeg'])

	show_file = st.empty()
	if source is not None:
		show_file.info("File received!")

	demo_image = r"C:\Users\vaish\Downloads\breast_self_examination.png"
	#st.image(demo_image, caption='Predict your diagnosis')
	image = cv2.imread(demo_image)
	st.image(image, caption='Image of a breast')
	tfile = tempfile.NamedTemporaryFile(suffix = '.png', delete=False)
	print("Analyzing your results....")

	# Checking if the file is being run as a script or imported as a module.
	if not source:
		vid = cv2.imread(demo_image)
		tfile.name = demo_image
		dem_img = open(tfile.name , 'rb')
		demo_bytes = dem_img.read()

		st.sidebar.text("Input Image")
		#st.sidebar.image(demo_bytes)       
	else:
		tfile.write(source.read())
		dem_vid = open(tfile.name , 'rb')
		demo_bytes = dem_vid.read()
		#st.text(demo_bytes)
		st.sidebar.text("Input Image")
		st.sidebar.text("Please give Histhopathological Image only")

	print(tfile.name)

	Start = st.sidebar.button('Get your results')
	stop = st.sidebar.button('Stop')

	if Start: 
		ab = tfile.name
		#st.text(ab)
		x=run(source=ab)
		st.text(run(source=ab))
		if(x=="Defect"):
			st.text(" There is a Possibilty of Cancer in the Tissue")
		else:
			st.text("No Cancer Detected")

		st.text("The Analysis may not be 100% accurate, Recommend consulting HCP for proper diagnosis!! ")

	if stop:
		Start = False
		print(Start)
		st.text("Processing has ended. You may close the tab now.")

if __name__ == '__main__':
	try:
		gui()
	except SystemExit:
		pass
