from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo y scaler
with open('models/model_svc.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Orden de las columnas (de tu dataset)
COLUMNAS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    datos = []
    for col in COLUMNAS:
        datos.append(float(request.form[col]))
    
    # Escalar y predecir
    datos_escalados = scaler.transform([datos])
    prediccion = model.predict(datos_escalados)[0]
    proba = model.predict_proba(datos_escalados)[0]
    
    if prediccion == 1:
        resultado = "⚠️ ALTO RIESGO de enfermedad cardíaca"
        color = "red"
        prob_text = f"Riesgo: {proba[1]*100:.1f}%"
    else:
        resultado = "✅ BAJO RIESGO de enfermedad cardíaca"
        color = "green"
        prob_text = f"Confianza: {proba[0]*100:.1f}%"
    
    return render_template('resultado.html', 
                         resultado=resultado,
                         probabilidad=prob_text,
                         color=color)

if __name__ == '__main__':
    app.run(debug=True, port=5000)