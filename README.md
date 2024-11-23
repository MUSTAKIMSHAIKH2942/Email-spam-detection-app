## **Email Spam Detection App**

A **Machine Learning-powered application** for classifying messages (emails or SMS) as **spam** or **ham (not spam)**. This app is built using **Python**, **Scikit-learn**, and **Streamlit** to provide an interactive and user-friendly interface for spam detection.

---

## **Features**
- Pre-trained **Naive Bayes classifier** for spam detection.
- Interactive UI for single-message classification.
- Preprocessing pipeline for text cleaning and vectorization.
- Easy-to-use web interface with real-time predictions.

---

## **Demo**
Here are some example inputs to test the app:

| Input Example                                                      | Predicted Output | Example Image                         |
|---------------------------------------------------------------------|------------------|---------------------------------------|
| "Hi, are we still on for the meeting tomorrow?"                    | **Ham**          | ![Ham Example](https://github.com/MUSTAKIMSHAIKH2942/Email-spam-detection-app/blob/main/project_spam/spam-detection/img/Captureoutputing.JPG) |
| "Congratulations! You've won a $1,000 gift card. Click to claim!"  | **Spam**         | ![Spam Example](https://github.com/MUSTAKIMSHAIKH2942/Email-spam-detection-app/blob/main/project_spam/spam-detection/img/spamoutput.JPG) |

---

## **Installation**
### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detection-app.git
   cd spam-detection-app


## Install the required packages:

pip install -r requirements.txt

## Run the Streamlit app:

streamlit run app.py
Open the app in your browser (usually at http://localhost:8501).

## Project Structure

 ![Project Structure Example](https://github.com/MUSTAKIMSHAIKH2942/Email-spam-detection-app/blob/main/project_spam/spam-detection/img/runningapp.JPG) |


How It Works

Preprocessing: Input text is cleaned (removal of stopwords, punctuation, etc.) and transformed into numerical features using TF-IDF vectorization.
Classification: A Naive Bayes model predicts whether the input text is spam or ham.
Interactive App: Streamlit provides an easy-to-use UI where users can input text for classification.

## Dataset
The dataset used in this project is a collection of labeled SMS and email messages.

Columns:
v1: Label (ham for not spam, spam for spam)
v2: Message content
Example:
csv
v1,v2
ham,"Hi there, how are you?"
spam,"Congratulations! You've won a prize."


## Accuracy: 0.9739910313901345

## Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.98      0.98       965
           1       0.88      0.94      0.91       150

    accuracy                           0.97      1115
   macro avg       0.93      0.96      0.95      1115
weighted avg       0.98      0.97      0.97      1115

## Technologies Used
Programming Language: Python
Libraries:
pandas: For data manipulation
scikit-learn: For ML model training
joblib: For saving/loading models
Streamlit: For building the interactive UI


## Future Enhancements
Add support for batch classification by uploading files (e.g., CSV).
Provide a confidence score for each prediction.
Train with larger and more diverse datasets for improved accuracy.

## Contributing
Contributions are welcome! If you'd like to improve the app, feel free to fork the repository and submit a pull request.

