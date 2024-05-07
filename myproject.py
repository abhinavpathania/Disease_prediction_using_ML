import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import tkinter as tk

# Load your dataset
data = pd.read_csv("DATA.csv")

# Separate features (X) and target labels (y)
X = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]  # target

# Initialize LabelEncoders for features and target labels
feature_label_encoders = []
for column in X.columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    feature_label_encoders.append(le)

# Encode target labels
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine Classifier
clf = SVC()
clf.fit(X_train, y_train)

# Implement cross-validation
cross_val_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

# Get the accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Initialize Tkinter GUI
class DiseasePredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Disease Prediction App")

        # Set the size of the window
        master.geometry("550x300")  # Width x Height

        # Create GUI components (labels, entry fields, buttons, etc.)
        self.label = tk.Label(master, text="Enter your symptoms (comma-separated):")
        self.label.pack(pady=10)

        self.entry = tk.Entry(master)
        self.entry.pack(pady=10)

        self.button = tk.Button(master, text="Predict", command=self.predict_disease)
        self.button.pack(pady=10)

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.accuracy_label = tk.Label(master, text=f"Model Accuracy: {accuracy:.2%} (Cross-Validation: {cross_val_accuracy:.2%})", font=("Helvetica", 12, "bold"))
        self.accuracy_label.pack(pady=10)

        self.warning_label = tk.Label(master, text="Warning: This application provides predictions based on machine learning and data collected from \nthe internet. The predictions are not a substitute for professional medical advice.\nConsult a doctor for accurate diagnosis and treatment.")
        self.warning_label.pack(side="bottom", pady=10)

    def predict_disease(self):
        # Get user input from the entry field
        user_input = self.entry.get().split(',')

        try:
            if len(user_input) < 5:
                raise ValueError("Please enter at least 5 symptoms.")

            # Encode user input using the saved LabelEncoders
            user_input_encoded = [le.transform([symptom])[0] if symptom in le.classes_ else -1 for le, symptom in zip(feature_label_encoders, user_input)]

            # Check for unknown symptoms
            unknown_symptoms = [symptom for symptom, encoded_value in zip(user_input, user_input_encoded) if encoded_value == -1]

            if unknown_symptoms:
                unknown_symptoms_str = ', '.join(unknown_symptoms)
                self.result_label.config(text=f"Unknown symptoms detected: {unknown_symptoms_str}")
            else:
                # Make a prediction using the trained classifier
                predicted_disease = clf.predict([user_input_encoded])[0]

                # Transform the predicted label back to the original disease label
                predicted_disease_original = label_encoder_y.inverse_transform([predicted_disease])

                # Update the result label with the predicted disease
                self.result_label.config(text=f"Predicted Disease: {predicted_disease_original[0]}")
        except ValueError as e:
            self.result_label.config(text=f"Error: {e}")
        except Exception as e:
            self.result_label.config(text=f"An unexpected error occurred: {e}")

# Create the main application window
root = tk.Tk()
app = DiseasePredictionApp(root)

# Run the GUI application
root.mainloop()
