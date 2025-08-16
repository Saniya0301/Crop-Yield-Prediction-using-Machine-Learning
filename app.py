from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print("Scikit-learn version:", sklearn.__version__)

# Load models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))  # fixed spelling

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form values
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            # Prepare input
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            transformed_features = preprocessor.transform(features)

            # Predict
            prediction = dtr.predict(transformed_features)[0]

            return render_template('index.html', prediction=round(prediction, 2))

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
