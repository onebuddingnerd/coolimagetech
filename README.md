
File Objective Descriptions:

	1. image_manips1.py
		- Importing an image into the program (as a cv2 object) and displaying it
		- Importing an image into the program, converting it to greyscale, and then displaying it
		- Importing an image into the program and displaying it using the matplotlib UI

	2. video_capture1.py
		- Displaying camera feed
			- in both color and greyscale (by processing the frames individually, using the technique in image_manips1.py, before output)

	3. face_box.py
		- Using the HAAR cascade pre-trained classifier (open-source, link to github source .xml code in file) to detect faces on a camera feed
		- Drawing rectangles around the detections

----

Pending goal: implement custom classifiers to recognize specific people, rather than generically indicate the presence of a face.
	- Question: how do we engineer the dataset to train such a classifier?
	- Question: how do we write the classifier so that it's compatible with deployment in a script like face_box.py?


