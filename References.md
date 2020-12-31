
#### Login/Signup

1. I just picked the mean value across the three channels on the z axis and coded up the L2 Norm formula (mean of squared cell values) 
2. Using Advanced DL Techniques for more Robust Face Recognition:
	* https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
		* Requires PyTorch for making embeddings, install available here https://pytorch.org/
			* embeddings are vectors for the image that have been passed through a neural network to generate a lower-dimensional representation of the image that (hopefully) brings attention to differentiating features and is more readily applicable for classification/verification tasks
	* **Dlib** will enable pretty straightforward generation of embeddings, which should be markedly better for photo comparison, as shown here: https://medium.com/data-science-lab-amsterdam/face-recognition-with-python-in-an-hour-or-two-d271324cbeb3
		* Install: https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/

#### OpenCV Links:

* Getting Started: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
* Video Capture: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
* Lowdown on Cascade Classifiers: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html


**PS:** And, really, that's all the main stuff. See! I haven't really done that much when it's presented in condensed summary!