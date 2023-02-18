
## README Latest (Above the First Divider)

#### Description
	
This is a grocery list guesser with face-based authentication (for new-user signup and returning-user login) and gesture-based sentiment evaluation (for classifying the reaction to a guessed grocery item). These are accomplished with persistent local storage of data between program executions and the OpenCV library in Python3 (to execute fature-detection on images), respectively. Additionally, there is code for a neural network in PyTorch, trained on a Kaggle dataset (warning: dataset is NOT included in this repository) of millions of grocery store purchases at Aldi, which enables more robust prediction on users' future purchases given their purchase histories. Lastly, there is automation code, implemented using the Selenium library, that launches queries on the Giant Eagle webpage and parses output to extract the price of the queried item -- a functionality designed to facilitate budgeting.
	
#### Execute

Run the following command on the command line, once the repo is cloned/downloaded:
		
	python3 gui1-1.py


