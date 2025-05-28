

from flask import Flask, render_template_string, request, flash, render_template
import numpy as np
import os
import cv2
from keras.models import load_model
from pyngrok import ngrok


ngrok.set_auth_token("2xiEKpOOWfaSCZlHTS84EikYmYA_3MpRAfaWDWnQeEbGTdNVP")


app = Flask(__name__)
app.secret_key = 'emotion_detection_app'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
try:
   model = load_model('custom_cnn_model.h5')
   model_loaded = True
except Exception as e:
   print(f"Error loading model: {e}")
   model_loaded = False

# Emotion labels based on your model training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/', methods=['GET', 'POST'])
def index():
   prediction = None
   if request.method == 'POST':
       print("[DEBUG] POST request received")

       if not model_loaded:
           print("[ERROR] Model not loaded")
           flash("Model not loaded. Please ensure 'custom_cnn_model.h5' exists.")
           return render_template('index.html', prediction=None, error=True)

       if 'image' not in request.files:
           print("[ERROR] No image in request")
           flash('No file part')
           return render_template('index.html', prediction=None, error=True)

       file = request.files['image']
       if file.filename == '':
           print("[ERROR] No selected file")
           flash('No selected file')
           return render_template('index.html', prediction=None, error=True)

       if file:
           filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
           file.save(filepath)
           print(f"[INFO] File saved to {filepath}")

           try:
               # Load image and detect face
               img = cv2.imread(filepath)
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

               face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
               faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

               if len(faces) == 0:
                   print("[ERROR] No face detected in the image.")
                   flash('No face detected in the uploaded image.')
                   return render_template('index.html', prediction=None, error=True)

               # Use the first detected face
               (x, y, w, h) = faces[0]
               face = gray[y:y+h, x:x+w]
               face = cv2.resize(face, (48, 48))
               face = face.astype('float32') / 255.0
               face = np.reshape(face, (1, 48, 48, 1))

               prediction_idx = np.argmax(model.predict(face))
               prediction = emotion_labels[prediction_idx]
               print(f"[SUCCESS] Predicted Emotion: {prediction}")
               
           except Exception as e:
               print(f"[ERROR] Exception during processing: {str(e)}")
               flash(f'Error processing image: {str(e)}')
               return render_template('index.html', prediction=None, error=True)

   return render_template('index.html', prediction=prediction, error=False)

if __name__ == '__main__':
   if model_loaded:
       print("Model loaded successfully. Starting application...")
   else:
       print("WARNING: Model not loaded. Place 'custom_cnn_model.h5' in the root directory.")

   # Start ngrok tunnel
   public_url = ngrok.connect(5001)
   print(f" * ngrok tunnel available at: {public_url}")

   app.run(port=5001)





from flask import Flask, render_template_string, request, flash, render_template
import numpy as np
import os
import cv2
from keras.models import load_model

app = Flask(__name__)
app.secret_key = 'emotion_detection_app'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
try:
    model = load_model('custom_cnn_model.h5')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Emotion labels based on your model training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        print("[DEBUG] POST request received")

        if not model_loaded:
            print("[ERROR] Model not loaded")
            flash("Model not loaded. Please ensure 'custom_cnn_model.h5' exists.")
            return render_template('index.html', prediction=None, error=True)

        if 'image' not in request.files:
            print("[ERROR] No image in request")
            flash('No file part')
            return render_template('index.html', prediction=None, error=True)

        file = request.files['image']
        if file.filename == '':
            print("[ERROR] No selected file")
            flash('No selected file')
            return render_template('index.html', prediction=None, error=True)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"[INFO] File saved to {filepath}")

            try:
                # Load image and detect face
                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    print("[ERROR] No face detected in the image.")
                    flash('No face detected in the uploaded image.')
                    return render_template('index.html', prediction=None, error=True)

                # Use the first detected face
                (x, y, w, h) = faces[0]
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.reshape(face, (1, 48, 48, 1))

                prediction_idx = np.argmax(model.predict(face))
                prediction = emotion_labels[prediction_idx]
                print(f"[SUCCESS] Predicted Emotion: {prediction}")
                
            except Exception as e:
                print(f"[ERROR] Exception during processing: {str(e)}")
                flash(f'Error processing image: {str(e)}')
                return render_template('index.html', prediction=None, error=True)

    return render_template('index.html', prediction=prediction, error=False)

if __name__ == '__main__':
    if model_loaded:
        print("Model loaded successfully. Starting application...")
    else:
        print("WARNING: Model not loaded. Place 'custom_cnn_model.h5' in the root directory.")

    app.run(port=5001)
