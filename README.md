# AI-Personality-Detection
Step 1: Clone the Repository
Clone this repository to your local machine using Git. In your terminal or command prompt, run:

bash
Copy
Edit
git clone https://github.com/yourusername/AI-Personality-Detection.git
Step 2: Create a Virtual Environment
It’s recommended to use a virtual environment to manage the dependencies for this project.

Navigate to your project folder:

bash
Copy
Edit
cd AI-Personality-Detection
Create a virtual environment:

For Windows:

bash
Copy
Edit
python -m venv venv
For macOS/Linux:

bash
Copy
Edit
python3 -m venv venv
Activate the virtual environment:

Windows:

bash
Copy
Edit
venv\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Step 3: Install Dependencies
Now that the virtual environment is activated, you can install all the required dependencies.

Create a requirements.txt file if not already present in your repo, and add the necessary libraries:

txt
Copy
Edit
pandas
scikit-learn
joblib
Install the dependencies from the requirements.txt file:

bash
Copy
Edit
pip install -r requirements.txt
Alternatively, you can manually install the dependencies using:

bash
Copy
Edit
pip install pandas scikit-learn joblib
Step 4: Run the Project
Once everything is set up and dependencies are installed, you can now train the model using the train_model.py script.

Run the training script to train the model:

bash
Copy
Edit
python train_model.py
After running the script, the model and vectorizer will be saved in the models/ directory.

Step 5: Use the Trained Model for Prediction
To use the saved model and vectorizer for making predictions, you can load them with the following code:

python
Copy
Edit
import joblib

# Load the model and vectorizer
model = joblib.load('models/personality_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Example text input
text = ["I enjoy socializing with friends."]
X = vectorizer.transform(text)
prediction = model.predict(X)
print(prediction)
