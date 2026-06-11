from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model once at startup
model = joblib.load('rf_model.pkl')

# --- Label encoding mappings (must match training-time encoding) ---
GENDER_MAP = {'Male': 1, 'Female': 0}
FEVER_MAP  = {'Yes': 1, 'No': 0}
COUGH_MAP  = {'Yes': 1, 'No': 0}

# City encoding: LabelEncoder sorts alphabetically; update if your training set differs
# Based on typical covid_toy dataset cities
CITY_MAP = {
    'Bangalore': 0,
    'Kolkata': 1,
    'Mumbai': 2,
    'Delhi': 3,
}


@app.route('/')
def index():
    return render_template('index.html', city_list=sorted(CITY_MAP.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        # Accept either raw numeric list OR structured form input
        if 'input' in data:
            input_values = data['input']
            if not isinstance(input_values, list):
                return jsonify({'error': 'input must be a list'}), 400
        else:
            # Structured input from the frontend form
            age    = data.get('age')
            gender = data.get('gender')
            fever  = data.get('fever')
            cough  = data.get('cough')
            city   = data.get('city')

            missing = [k for k, v in [('age', age), ('gender', gender),
                                       ('fever', fever), ('cough', cough),
                                       ('city', city)] if v is None]
            if missing:
                return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

            try:
                age = int(age)
            except (ValueError, TypeError):
                return jsonify({'error': 'age must be an integer'}), 400

            gender_enc = GENDER_MAP.get(gender)
            fever_enc  = FEVER_MAP.get(fever)
            cough_enc  = COUGH_MAP.get(cough)
            city_enc   = CITY_MAP.get(city)

            if any(v is None for v in [gender_enc, fever_enc, cough_enc, city_enc]):
                return jsonify({'error': 'Invalid value for gender, fever, cough, or city'}), 400

            input_values = [age, gender_enc, fever_enc, cough_enc, city_enc]

        input_data = np.array(input_values, dtype=float).reshape(1, -1)

        if input_data.shape[1] != model.n_features_in_:
            return jsonify({
                'error': f'Expected {model.n_features_in_} features, got {input_data.shape[1]}'
            }), 400

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        covid_prob = float(probability[1]) if len(probability) > 1 else float(probability[0])

        result = 'Positive' if int(prediction[0]) == 1 else 'Negative'
        return jsonify({
            'prediction': int(prediction[0]),
            'result': result,
            'confidence': round(covid_prob * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
