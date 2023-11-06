import tensorflow as tf
import numpy as np
class_names = ['glass','plastic']
# Ruta a la imagen local (reemplaza 'ruta_de_la_imagen_local.jpg' con la ruta real)
<<<<<<< HEAD
imagen_local = "prove3.jpeg"
=======
imagen_local = 'glass6.jpg'
>>>>>>> 7e559c27b2fc30134c10196f6e7a9b6228e98253
img_height = 180
img_width = 180
# Cargar la imagen local
img = tf.keras.utils.load_img(imagen_local, target_size=(img_height, img_width))


img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

TF_LITE_MODEL_FILE_PATH = 'model.tflite'  # Ruta al archivo TFLite
interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL_FILE_PATH)
interpreter.allocate_tensors()

# Obtener los detalles de la entrada y la salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Establecer la entrada del modelo
interpreter.set_tensor(input_details[0]['index'], img_array)

# Ejecutar la inferencia
interpreter.invoke()

# Obtener la salida del modelo
output_data = interpreter.get_tensor(output_details[0]['index'])

# Realizar la clasificación
score_lite = tf.nn.softmax(output_data)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)