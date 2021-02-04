
## README Latest (Above the First Divider)

#### Description
	
This is a grocery list guesser with face-based authentication (for new-user signup and returning-user login) and gesture-based sentiment evaluation (for classifying the reaction to a guessed grocery item). These are accomplished with persistent local storage of data between program executions and the OpenCV library in Python3 (to execute fature-detection on images), respectively. Additionally, there is code for a neural network in PyTorch, trained on a Kaggle dataset (warning: dataset is NOT included in this repository) of millions of grocery store purchases at Aldi, which enables more robust prediction on users' future purchases given their purchase histories. Lastly, there is automation code, implemented using the Selenium library, that launches queries on the Giant Eagle webpage and parses output to extract the price of the queried item -- a functionality designed to facilitate budgeting.
	
#### Execute

Run the following command on the command line, once the repo is cloned:
		
	python3 gui1-1.py

----

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

#### Pending Intents:

1. Cool applications based on storing user data and presenting content tailored to users (based on that data)

#### To-Do
1. Instead of printing to terminal, show recommendations on GUI
2. Get total price and show on GUI
3. Show neural network
4. Hand recognition - need to have video in layout


