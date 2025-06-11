import os
from flask import Flask, request, render_template
import numpy as np
import cv2
import joblib
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ✅ Load pipeline (includes scaler + model)
MODEL_PATH = 'iris_pipeline.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model pipeline loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
    model = None
except Exception as e:
    print(f"❌ Error loading model pipeline: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_iris_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        if cv2.countNonZero(thresh) == 0:
            _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        default_iris_features_cm = np.array([5.01, 3.42, 1.46, 0.24])  # Setosa average

        if len(contours) < 2:
            print("⚠️ Not enough contours found. Returning default features.")
            return default_iris_features_cm

        # Sort contours by area
        selected_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Assume largest = sepal, second = petal
        rect_sepal = cv2.minAreaRect(selected_contours[0])
        (_, _), (w_sepal, h_sepal), _ = rect_sepal
        sepal_length = max(w_sepal, h_sepal) * 0.027
        sepal_width  = min(w_sepal, h_sepal) * 0.027

        rect_petal = cv2.minAreaRect(selected_contours[1])
        (_, _), (w_petal, h_petal), _ = rect_petal
        petal_length = max(w_petal, h_petal) * 0.027
        petal_width  = min(w_petal, h_petal) * 0.027

        final_features_cm = np.array([
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ])

        print("👉 Final extracted features (cm):", final_features_cm)
        return final_features_cm

    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return np.array([5.01, 3.42, 1.46, 0.24])  # fallback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', result="Error: Model pipeline not loaded.")

    if 'image' not in request.files:
        return render_template('result.html', result="No image file provided."), 400

    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', result="No selected file."), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        features = extract_iris_features(filepath)
        os.remove(filepath)

        if features is None:
            return render_template('result.html', result="Error: Could not extract features from image."), 500

        try:
            features_2d = features.reshape(1, -1)
            prediction_label = model.predict(features_2d)[0]
            prediction_proba = model.predict_proba(features_2d)[0]

            class_labels = model.classes_
            probabilities_str = ", ".join(
                [f"{class_labels[i]}: {prob:.2f}" for i, prob in enumerate(prediction_proba)]
            )

            result_message = (
                f"Predicted Flower Species: <strong>{prediction_label}</strong><br>"
                f"Raw Extracted Features (cm): {features.tolist()}<br>"
                f"Probabilities: {probabilities_str}"
            )
            return render_template('result.html', result=result_message)

        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return render_template('result.html', result=f"Prediction error: {e}")
    else:
        return render_template('result.html', result="Invalid file type."), 400

if __name__ == '__main__':
    app.run(debug=True)
