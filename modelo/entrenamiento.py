# ENTRENAMIENTO DE RED NEURONAL PARA PREDICCIÓN DE RIESGO ACADÉMICO
# Este código entrena una red neuronal para predecir si un estudiante:
# - Abandonará (Dropout)
# - Continuará activo (Enrolled)  
# - Se graduará (Graduate)


# Importamos librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("="*70)
print("Iniciando entrenamiento de red neuronal para predicción de riesgo académico")
print("="*70)


# 1. Cargamos y exploramos el dataset

print("\n[1/6] Cargando dataset...")
df = pd.read_csv('dataset.csv', sep=';')
print(f"✓ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# Verificar valores faltantes
if df.isnull().sum().sum() > 0:
    print("  → Limpiando valores faltantes...")
    df = df.dropna()
    print(f"  → Registros después de limpieza: {df.shape[0]}")

# Mostrar distribución de clases
print("\n  Distribución de clases:")
target_counts = df['Target'].value_counts()
for clase, cantidad in target_counts.items():
    porcentaje = (cantidad / len(df)) * 100
    print(f"    - {clase}: {cantidad} ({porcentaje:.1f}%)")


# 2. Preparacion de características 

print("\n[2/6] Preparando características...")

# Crear características derivadas que coinciden con las del frontend
df['age'] = df['Age at enrollment']
df['admission_grade'] = df['Admission grade']
df['scholarship'] = df['Scholarship holder']

# Calcular cursos aprobados totales (suma de ambos semestres)
df['approved'] = (df['Curricular units 1st sem (approved)'] + 
                  df['Curricular units 2nd sem (approved)'])

# Calcular promedio académico (promedio de ambos semestres)
# Manejar casos donde el promedio es 0
grade_1st = df['Curricular units 1st sem (grade)']
grade_2nd = df['Curricular units 2nd sem (grade)']

# Calcular promedio ponderado considerando solo semestres con notas
df['avg_grade'] = np.where(
    (grade_1st > 0) & (grade_2nd > 0),
    (grade_1st + grade_2nd) / 2,
    np.where(grade_1st > 0, grade_1st, 
             np.where(grade_2nd > 0, grade_2nd, 0))
)

# Seleccionar características para el modelo
caracteristicas = ['age', 'admission_grade', 'scholarship', 'approved', 'avg_grade']
X = df[caracteristicas].values

print(f"✓ Características seleccionadas: {caracteristicas}")
print(f"  Forma de X: {X.shape}")


# 3. Preparacion de etiquetas

print("\n[3/6] Codificando etiquetas...")

# Codificar las clases (Target) a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Target'])

# Mostrar mapeo de clases
print("  Mapeo de clases:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"    {i} → {clase}")

# Convertir a formato categórico
y_categorical = to_categorical(y_encoded, num_classes=3)
print(f"✓ Etiquetas codificadas: {y_categorical.shape}")


# 4. Normalización y división de datos

print("\n[4/6] Normalizando y dividiendo datos...")

# Normalizar características (importante para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento 80% y prueba 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # Mantenemos proporción de clases
)

print(f"✓ Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"✓ Datos de prueba: {X_test.shape[0]} muestras")


# 5. Construcción de la arquitectura de la red neuronal

print("\n[5/6] Construyendo arquitectura de la red neuronal...")

# Crear modelo secuencial
modelo = Sequential([
    # Capa de entrada (5 características)
    Dense(units=64, activation='relu', input_shape=(5,), name='capa_entrada'),
    Dropout(0.3),  # Regularización para evitar overfitting
    
    # Primera capa oculta
    Dense(units=128, activation='relu', name='capa_oculta_1'),
    Dropout(0.3),
    
    # Segunda capa oculta
    Dense(units=64, activation='relu', name='capa_oculta_2'),
    Dropout(0.2),
    
    # Tercera capa oculta
    Dense(units=32, activation='relu', name='capa_oculta_3'),
    
    # Capa de salida (3 clases: Dropout, Enrolled, Graduate)
    Dense(units=3, activation='softmax', name='capa_salida')
])

# Compilamos el modelo
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Para clasificación multiclase
    metrics=['accuracy']
)

print("Arquitectura del modelo:")
modelo.summary()


# 6. Entrenamiento del modelo

print("\n[6/6] Entrenando el modelo...")
print("  (Esto puede tardar unos minutos...)")

# Configuramos early stopping para evitar sobreentrenamiento
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Entrenar el modelo
historial = modelo.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # 20% de datos de entrenamiento para validación
    callbacks=[early_stop],
    verbose=2
)

print("\n✓ Entrenamiento finalizado")


# 7. Evaluación del modelo

print("\n" + "="*70)
print("EVALUACIÓN DEL MODELO")
print("="*70)

# Evaluar en conjunto de prueba
loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\n✓ Precisión en conjunto de prueba: {accuracy*100:.2f}%")
print(f"✓ Pérdida en conjunto de prueba: {loss:.4f}")

# Hacer predicciones de ejemplo
print("\n--- Predicciones de ejemplo ---")
y_pred = modelo.predict(X_test[:5], verbose=0)
for i in range(5):
    clase_predicha = np.argmax(y_pred[i])
    clase_real = np.argmax(y_test[i])
    confianza = np.max(y_pred[i]) * 100
    print(f"Ejemplo {i+1}: Predicción={label_encoder.classes_[clase_predicha]} "
          f"({confianza:.1f}% confianza), Real={label_encoder.classes_[clase_real]}")

# 8. Guardamos el modelo

print("\n" + "="*70)
print("GUARDANDO MODELO")
print("="*70)

# Guardamos el modelo en formato .h5
modelo_path = '../backend/modelo_ia.h5'
modelo.save(modelo_path)
print(f"✓ Modelo guardado exitosamente en: {modelo_path}")

# Guardamos también los parámetros del scaler para uso futuro (opcional)
import pickle
scaler_path = '../backend/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler guardado en: {scaler_path}")


# 9. Visualizmos gráficas de entrenamiento

print("\n" + "="*70)
print("Generando gráficas")
print("="*70)

# Crear gráficas del entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Gráfica de precisión
ax1.plot(historial.history['accuracy'], label='Entrenamiento', linewidth=2)
ax1.plot(historial.history['val_accuracy'], label='Validación', linewidth=2)
ax1.set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Precisión')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica de pérdida
ax2.plot(historial.history['loss'], label='Entrenamiento', linewidth=2)
ax2.plot(historial.history['val_loss'], label='Validación', linewidth=2)
ax2.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Pérdida')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('entrenamiento_resultados.png', dpi=300, bbox_inches='tight')
print("✓ Gráficas guardadas en: entrenamiento_resultados.png")


# 10. Resumen final del entrenamiento

print("\n" + "="*70)
print("Resumen del entrenamiento")
print("="*70)
print(f"✓ Total de muestras procesadas: {len(df)}")
print(f"✓ Características utilizadas: {len(caracteristicas)}")
print(f"✓ Clases predichas: 3 (Dropout, Enrolled, Graduate)")
print(f"✓ Precisión final: {accuracy*100:.2f}%")
print(f"✓ Épocas entrenadas: {len(historial.history['loss'])}")
print(f"✓ Modelo guardado: {modelo_path}")
print("\n" + "="*70)
print("¡Entrenamiento completado exitosamente!")
print("="*70)
print("\nEl modelo está listo para ser usado por el backend Flask.")
print("Para iniciar el servidor: python ../backend/app.py")