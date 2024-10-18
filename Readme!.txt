Running the Subsidy Prediction Application

Step 1: Install Required Dependencies
Before starting the application ensure that all required packages are installed. Use the following command to install dependencies listed in requirements.txt:

pip install -r requirements.txt

Step 2: (Optional) Retrain the Model and Label Encoders
If you need to retrain the model or label encoders, follow these instructions. However, if you are using the pre-trained models and encoders (already provided) you can skip this step.

Open the Train_models_Encoders.ipynb notebook.
Run the notebook until you see the message:
Model and LabelEncoders saved successfully!
This will retrain and save the model and label encoders for future use.

Step 3: Start the Website
Open the predictor.py file.
Run the file using the following command:
python predictor.py or use PyCharm/visual studio like program to run the file.

After running the command/file, you will receive a link to access the subsidy prediction website (usually http://localhost:XXXX).

Notes!!!!!:
You might encounter some errors related to pandas while running the application.
These errors can be safely ignored as they do not affect the functionality of the website.