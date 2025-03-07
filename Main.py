from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and configure labels
model = load_model('model/best_model.keras')
all_labels = [
    'Apple___Apple_scab', 
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy', 
    'Cherry_(including_sour)_Powdery_mildew',
    'Cherry_(including_sour)_healthy',
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)Common_rust',
    'Corn_(maize)_Northern_Leaf_Blight',
    'Corn_(maize)_healthy',   
    'Grape___Black_rot',
    'Grape__Esca(Black_Measles)',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,bell__Bacterial_spot',
    'Pepper,bell__healthy',     
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',    
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',   
]

treatments = {
    "Apple___Apple_scab": "Treatment: Apply fungicides like captan or mancozeb during the growing season and practice proper orchard sanitation to control Apple Scab.",
        "Apple___Black_rot": "Treatment: Treat Apple Black Rot by pruning and removing infected branches, applying fungicides like captan or mancozeb during the growing season, and maintaining proper orchard sanitation.",
        "Apple___Cedar_apple_rust": "Treatment: Apply a fungicide containing myclobutanil or propiconazole during the early stages of apple tree growth to manage Cedar Apple Rust.",
        "Apple___healthy": "Treatment: No treatment required; crop is healthy.",
        "Blueberry___healthy": "Treatment: No treatment required; crop is healthy.",
        "Cherry_(including_sour)_Powdery_mildew": "Treatment: Treat Cherry Powdery Mildew with fungicides like sulfur or myclobutanil, prune infected areas, and improve air circulation around the trees.",
        "Cherry_(including_sour)_healthy": "Treatment: No treatment required; plant is healthy.",
        "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot": "Treatment: Treat Corn Gray Leaf Spot with fungicides like strobilurins or triazoles, plant resistant hybrids, and practice crop rotation with non-host crops.",
        "Corn_(maize)Common_rust": "Treatment: Treat Corn Common Rust with fungicides like azoxystrobin or tebuconazole, plant resistant hybrids, and ensure timely planting to reduce disease severity.",
        "Corn_(maize)_Northern_Leaf_Blight": "Treatment: Treat Corn Northern Leaf Blight by applying fungicides like strobilurins or triazoles, planting resistant hybrids, and practicing crop rotation and residue management.",
        "Corn_(maize)_healthy": "Treatment: No treatment required; crop is healthy.",
        "Grape___Black_rot": "Treatment: Apply fungicides such as Mancozeb or Captan; prune infected parts to improve airflow.",
        "Grape__Esca(Black_Measles)": "Treatment: Remove and destroy infected wood; apply fungicides containing Trifloxystrobin.",
        "Grape__Leaf_blight(Isariopsis_Leaf_Spot)": "Treatment: Use fungicides like Copper-based sprays; improve ventilation in the vineyard.",
        "Grape___healthy": "Treatment: No treatment required; crop is healthy.",
        "Orange__Haunglongbing(Citrus_greening)": "Treatment: Remove and destroy infected trees; use disease-free saplings and control psyllid vectors.",
        "Peach___Bacterial_spot": "Treatment: Apply copper-based bactericides; avoid overhead irrigation and prune infected branches.",
        "Peach___healthy": "Treatment: No treatment required; crop is healthy.",
        "Pepper,bell__Bacterial_spot": "Treatment: Apply copper-based fungicides or bactericides; rotate crops and use resistant varieties.",
        "Pepper,bell__healthy": "Treatment: No treatment required; crop is healthy.",
        "Potato___Early_blight": "Treatment: Treat Potato Early Blight with fungicides like Mancozeb or Chlorothalonil, ensure proper crop rotation, and maintain good plant health through balanced fertilization.",
        "Potato___Late_blight": "Treatment: Treat Potato Late Blight with systemic fungicides like Metalaxyl or Ridomil, practice crop rotation, and use certified disease-free seeds.",
        "Potato___healthy": "Treatment: No treatment required; plant is healthy.",
        "Raspberry___healthy": "Treatment: No treatment required; plant is healthy.",
        "Soybean___healthy": "Treatment: No treatment required; plant is healthy.",
        "Squash___Powdery_mildew": "Treatment: Treat Squash Powdery Mildew with fungicides like sulfur, potassium bicarbonate, or myclobutanil, and improve air circulation by proper spacing and pruning.",
         "Strawberry___Leaf_scorch": "Treatment: Treat Strawberry Leaf Scorch by removing infected leaves, applying fungicides like Captan or Myclobutanil, and ensuring proper irrigation and air circulation.",
        "Strawberry___healthy": "Treatment: No treatment required; plant is healthy.",
        "Tomato___Bacterial_spot": "Treatment: Treat Tomato Bacterial Spot by applying copper-based bactericides, removing infected plants, and practicing crop rotation with non-host crops.",
        "Tomato___Early_blight": "Treatment: Treat Tomato Early Blight with fungicides like Mancozeb or Chlorothalonil, remove infected leaves, and practice crop rotation and proper plant spacing.",
        "Tomato___Late_blight": "Treatment: Apply fungicides containing copper or chlorothalonil and remove infected plants to prevent the spread of Tomato Late Blight.",
        "Tomato___Leaf_Mold": "Treatment: Use a fungicide containing copper or chlorothalonil, improve air circulation by pruning, and avoid overhead irrigation to reduce humidity.",
        "Tomato___Septoria_leaf_spot": "Treatment: Remove and destroy infected leaves, improve air circulation, and apply a fungicide containing chlorothalonil or copper-based compounds for effective control.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Treatment: Use insecticidal soap or neem oil to control spider mites, and promote natural predators like ladybugs and predatory mites for long-term management.",
        "Tomato___Target_Spot": "Treatment: Use fungicides like Difenoconazole or Mancozeb, ensure proper spacing for airflow, and remove infected plant debris to prevent the spread of Tomato Target Spot.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Treatment: Plant resistant tomato varieties, control whitefly populations with insecticides like neem oil, and use reflective mulches to reduce virus spread.",
        "Tomato___Tomato_mosaic_virus": "Treatment: Remove and destroy infected plants immediately, practice crop rotation, and use resistant tomato varieties to prevent Tomato Mosaic Virus.",
        "Tomato___healthy": "Treatment: No treatment required; crop is healthy."
}

def simulate_disease_detection(image):
    """Simulating a segmentation result (binary mask where 1 represents diseased areas)."""
    return np.random.randint(0, 2, (256, 256), dtype=np.uint8)

def calculate_severity(mask):
    """Calculate the severity of disease based on mask."""
    total_pixels = mask.size
    diseased_pixels = np.sum(mask)
    severity_percentage = (diseased_pixels / total_pixels) * 100
    return severity_percentage

def severity_category(severity):
    """Map severity percentage to disease severity category."""
    if severity == 0:
        return "Healthy"
    elif severity <= 25:
        return "Mild"
    elif severity <= 50:
        return "Moderate"
    elif severity <= 75:
        return "Severe"
    else:
        return "Very Severe"

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home route
@app.route('/')
def crewwelcome():
    return render_template('crewwelcome.html')

@app.route('/crewlogin', methods=['GET', 'POST'])
def crewlogin():
    if request.method == 'POST':
        name = request.form.get('name')
        password = request.form.get('password')

        # Authentication logic
        if name and password:
            return redirect(url_for('crewextra1'))  # Redirect to the dashboard or any other page
        else:
            return render_template('crewlogin.html', error="Invalid credentials")
    return render_template('crewlogin.html')

@app.route('/crewextra1')
def crewextra1():
    return render_template('crewextra1.html')

@app.route('/crewextra2')
def crewextra2():
    return render_template('crewextra2.html')

@app.route('/crewextra3')
def crewextra3():
    return render_template('crewextra3.html')

@app.route('/upload', methods=['GET', 'POST'])
def crewuploaddummy():
    if request.method == 'POST':
        # Handle the file upload
        if 'image' not in request.files:
            return render_template('crewuploaddummy.html', result=None, error="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('crewuploaddummy.html', result=None, error="No file selected")

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Image processing
            img = Image.open(file_path).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_label = all_labels[np.argmax(predictions)]
            confidence = round(np.max(predictions) * 100, 2)
            treatment = treatments.get(predicted_label, "No treatment available")

            mask = simulate_disease_detection(img)
            severity = calculate_severity(mask)
            severity_category_label = severity_category(severity)

            return render_template(
                'crewuploaddummy.html',
                result={
                    "label": predicted_label,
                    "confidence": confidence,
                    "treatment": treatment,
                    "severity": severity,
                    "severity_category": severity_category_label
                },
                error=None
            )
        except Exception as e:
            return render_template('crewuploaddummy.html', result=None, error=str(e))
    return render_template('crewuploaddummy.html', result=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)