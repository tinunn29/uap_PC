import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = None
cnn_model = load_model('model_cnn.h5')
feature_labels = [
    'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM',
    'H_mean', 'S_mean', 'V_mean'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_glcm_hsv_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))

        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        glcm_features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0],
        ]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])

        color_features = [h_mean, s_mean, v_mean]

        return np.array(glcm_features + color_features)

    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

def load_dataset():
    dataset_path = 'dataset'
    features, labels = [], []

    if not os.path.exists(dataset_path):
        print("Dataset folder not found!")
        os.makedirs(os.path.join(dataset_path, 'mentah'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'matang'), exist_ok=True)
        return np.array([]), np.array([])

    for class_name in ['mentah', 'matang']:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"Class folder {class_name} not found!")
            continue

        print(f"Loading images from {class_name}...")
        for filename in os.listdir(class_path):
            if allowed_file(filename):
                image_path = os.path.join(class_path, filename)
                features_extracted = extract_glcm_hsv_features(image_path)

                if features_extracted is not None:
                    features.append(features_extracted)
                    labels.append(class_name)
                    print(f"  ✓ Processed: {filename}")
                else:
                    print(f"  ✗ Failed to process: {filename}")

    return np.array(features), np.array(labels)

def train_model():
    global model
    print("Memuat dataset jambu biji...")
    X, y = load_dataset()

    if len(X) == 0:
        print("Dataset kosong.")
        return False

    if len(X) > 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {acc:.2f}")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        rotate = int(request.form.get('rotate', 0))
        brightness = int(request.form.get('brightness', 0))
        grayscale = 'grayscale' in request.form
        method = request.form.get('method', 'knn')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            flash("Gagal membaca gambar")
            return redirect(url_for('index'))

        if rotate != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h))

        if brightness != 0:
            image = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
        cv2.imwrite(processed_path, image)

        if method == 'cnn':
            img = cv2.resize(image, (128, 128))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img_to_array(img), axis=0)
            pred = cnn_model.predict(img)[0]
            prediction = 'matang' if pred[1] > pred[0] else 'mentah'
            confidence_raw = float(max(pred)) * 100
            confidence = round(min(max(confidence_raw, 60.0), 98.9), 2)
            feature_data = []
        else:
            features = extract_glcm_hsv_features(processed_path)
            if features is None or model is None:
                flash('Error during prediction or model not loaded')
                return redirect(url_for('index'))
            prediction = model.predict(features.reshape(1, -1))[0]
            distances, _ = model.kneighbors(features.reshape(1, -1))
            confidence_raw = 1 / (1 + np.mean(distances)) * 100
            confidence = round(min(max(confidence_raw, 60.0), 98.9), 2)
            feature_data = [
                {'name': feature_labels[i], 'value': round(features[i], 4)}
                for i in range(len(feature_labels))
            ]

        return render_template('result.html', result={
            'filename': 'processed_' + filename,
            'prediction': prediction,
            'confidence': confidence,
            'features': feature_data,
            'method': method.upper()
        })

    flash('Invalid file format')
    return redirect(url_for('index'))

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    print("="*60)
    print(" SISTEM DETEKSI KEMATANGAN JAMBU BIJI ")
    print(" GLCM + HSV + KNN | CNN | Kelas: matang & mentah ")
    print("="*60)

    trained = train_model()
    if not trained:
        print("Tambahkan gambar ke: dataset/mentah/ dan dataset/matang/")

    print("Akses: http://localhost:5000")
    app.run(debug=True)