
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

Pending goal (in face_box_w_profile_saver_attempt.py): implement custom classifiers to recognize specific people, rather than generically indicate the presence of a face

	1. Right now, the buggy code can save an image of a person once a face is detected on the camera feed (in the "saved_faces" directory)
	
	2. The next objective is to get the program to compute a "similarity score" between a saved image and a detected face, if there is at least one face already stored
		a. Currently, it (erroneously) always ends up storing a new face and never running the similarity score code!


