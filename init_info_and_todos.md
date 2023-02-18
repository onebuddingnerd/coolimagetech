#### File Objective Descriptions:

1. `image_manips1.py`
	* Importing an image into the program (as a cv2 object) and displaying it
	* Importing an image into the program, converting it to greyscale, and then displaying it
	* Importing an image into the program and displaying it using the matplotlib UI

2. `video_capture1.py`
	* Displaying camera feed
		* in both color and greyscale (by processing the frames individually, using the technique in `image_manips1.py`, before output)

3. `face_box.py`
	* Using the HAAR cascade pre-trained classifier (open-source, link to github source .xml code in file) to detect faces on a camera feed
	* Drawing rectangles around the detections

4. `face_box_w_profile_saver_attempt(2).py`
	* '2' is latest version
	* Compare the detected-face on camera feed to faces already seen (i.e. "registered users") in prior runs of the program
		* Develop "difference" scores between the face on the camera feed and the faces of the "registered users." Ask if the user is the same as the photo with the lowest "difference" score.

----

#### To-Do (1/25/21)
1. Instead of printing to terminal, show recommendations on GUI
2. Get total price and show on GUI
3. Show neural network
4. Hand recognition - need to have video in layout