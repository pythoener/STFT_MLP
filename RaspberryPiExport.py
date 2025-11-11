import os, tensorflow as tf, joblib, numpy as np, scipy.signal as ss

export_dir = os.path.join(os.getcwd(), "RaspyExport")
os.makedirs(export_dir, exist_ok=True)

print("Loading model & scalers …")
model = tf.keras.models.load_model('Model_L_1.h5')
feature_scaler = joblib.load('feature_scaler.pkl')   # <‑‑ change if .gz
label_scaler   = joblib.load('label_scaler.pkl')

print("Converting Keras → int8 TF‑Lite …")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open(os.path.join(export_dir, 'model_l_1_int8.tflite'), 'wb').write(conv.convert())

joblib.dump(feature_scaler, os.path.join(export_dir, 'feat_scaler.gz'))
joblib.dump(label_scaler,   os.path.join(export_dir, 'label_scaler.gz'))

sos = ss.butter(8, 4000/8000, 'low', output='sos')
np.save(os.path.join(export_dir, 'lpf_sos.npy'), sos)

print("\n✓  Done – files generated in", export_dir)
for f in os.listdir(export_dir):
    print("   ", f)