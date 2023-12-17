This is a model for predicting the probability of clients default, based on LightGBM model from Microsoft.
# Pre-trained model
A pre-trained model can be found in the model sub-directory
# Train the model
## Install dependencies
pip install -r requirements.txt
## Get the data
Download the data for training and target here: https://drive.google.com/drive/folders/1nIAlRXj7MSuf8cwUOCFfXtiyqiHysLUd?usp=sharing
Save the files with data for training in the data sub-directory
Save the file with target in the target sub-directory
## Run the training
python pipeline.py fit
This command by default takes care of all the training.
# Prediction of default
To get the prediction from the model do the following:
Save file with data for pridiction in the data-for-predictions sub-directory (there's an example of what it should look like)
Run python pipeline.py predict
# Model's metadata
To get model's metadata simlpy run:
python pipeline.py metadata
