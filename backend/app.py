# AQUÍ IMPORTO LAS LIBRERÍAS
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)


# Carga del modelo 

modelo = tf.keras.models.load_model("modelo_ia.h5")

# CARGA DEL SCALER (Para normalizar los datos igual que en el entrenamiento)
# ==================================================================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



# ENDPOINT DE PREDICCIÓN

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json
        # Esto lo hacemos ya que los datos que vienen desde el formlario son cadena de texto
        entrada = np.array([[
            float(data["age"]),
            float(data["admission_grade"]),
            float(data["scholarship"]),
            float(data["approved"]),
            float(data["avg_grade"])
        ]])

        # NORMALIZAR los datos con el mismo scaler usado en el entrenamiento
        entrada_normalizada = scaler.transform(entrada)

        # Hacer la predicción con datos normalizados
        pred = modelo.predict(entrada_normalizada)
        clase = int(np.argmax(pred))

        labels = [
            "Deserción académica (riesgo alto)",
            "Estudiante activo (riesgo medio)",
            "Egresando / Graduado (bajo riesgo)"
        ]

        return jsonify({
            "prediccion": labels[clase],
            "confianza": f"{round(float(np.max(pred)) * 100, 2)}%",
            "detalle": {
                "clase_modelo": int(clase),
                "interpretacion": labels[clase]
        }
})

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)