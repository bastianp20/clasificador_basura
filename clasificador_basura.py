import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Eliminar logs anteriores
for modelo in ["cnn"]:
    log_dir = os.path.join("logs", modelo)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Logs de {modelo} eliminados.")

# aquí se define la ruta del datasets
ruta_dataset = "realwaste"
tamano_img = 100

# datos
clases = sorted(os.listdir(ruta_dataset))
datos_entrenamiento = []

for etiqueta, clase in enumerate(clases):
    ruta_clase = os.path.join(ruta_dataset, clase)
    for archivo in os.listdir(ruta_clase):
        ruta_imagen = os.path.join(ruta_clase, archivo)
        try:
            imagen = cv2.imread(ruta_imagen)
            imagen = cv2.resize(imagen, (tamano_img, tamano_img))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(tamano_img, tamano_img, 1)
            datos_entrenamiento.append([imagen, etiqueta])
        except:
            print(f"Error cargando imagen: {ruta_imagen}")

np.random.shuffle(datos_entrenamiento)

x = np.array([i[0] for i in datos_entrenamiento]).astype("float32") / 255.0
y = np.array([i[1] for i in datos_entrenamiento])

# division de los datos 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=30,          
    width_shift_range=0.1,     
    height_shift_range=0.1,     
    zoom_range=0.2,             
    shear_range=0.2,            
    horizontal_flip=True       
)

datagen.fit(x_train)
data_gen_train = datagen.flow(x_train, y_train, batch_size=32)

# Callbacks
tensorboard_cb = TensorBoard(log_dir=os.path.join("logs", "cnn"))

print("Clases detectadas:", clases)
print("Total de clases:", len(clases))

# modelo convolucional
modeloCNN = models.Sequential([
    layers.Input(batch_input_shape=(None, tamano_img, tamano_img, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(clases), activation='softmax')
])


modeloCNN.compile(optimizer='adam'
                  , loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# entrenamiento del modelo 
modeloCNN.fit(
    data_gen_train,
    epochs=50,
    validation_data=(x_val, y_val),
    steps_per_epoch=int(np.ceil(len(x_train) / 32)),
    callbacks=[tensorboard_cb]
)

modeloCNN.save("modelo_cnn.h5")

