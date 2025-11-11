# export_models.py
import os, sys, numpy as np, joblib, tensorflow as tf, scipy.signal as ss

# ---------- Paths ----------
# Put exports under a subfolder named "export" next to this script
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(BASE_DIR, "export")
os.makedirs(export_dir, exist_ok=True)

MODEL_H5 = "Model_L_1.h5"
FEAT_SCALER_PKL = "feature_scaler.pkl"  # change to .gz if that's what you saved
LAB_SCALER_PKL  = "label_scaler.pkl"

# ---------- Load ----------
print("Loading model & scalers …")
model = tf.keras.models.load_model(MODEL_H5)
feat_scaler = joblib.load(FEAT_SCALER_PKL)
lab_scaler  = joblib.load(LAB_SCALER_PKL)

def save_bytes(path, blob):
    with open(path, "wb") as f: f.write(blob)
    print(f"  wrote {os.path.basename(path):28s} {len(blob)/1024:.1f} KiB")

# ---------- Variant A: FP32 ----------
print("\nConverting → FP32 …")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = conv.convert()
save_bytes(os.path.join(export_dir, "model_fp32.tflite"), tfl)

# ---------- Variant B: FP16 weights (float32 I/O) ----------
print("Converting → FP16 (weights) …")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
save_bytes(os.path.join(export_dir, "model_fp16.tflite"), tfl)

# ---------- Variant C: INT8 dynamic range (float32 I/O) ----------
print("Converting → INT8 (dynamic range) …")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
tfl = conv.convert()
save_bytes(os.path.join(export_dir, "model_int8_dynamic.tflite"), tfl)

# ---------- Representative dataset for full INT8 ----------
def rep_dataset(num=512, dim=336, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(num):
        x = rng.normal(0.0, 1.0, size=(1, dim)).astype(np.float32)
        yield [x]

# ---------- Variant D: INT8 full (float32 I/O) ----------
print("Converting → INT8 (full, float32 I/O) …")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: rep_dataset()
tfl = conv.convert()
save_bytes(os.path.join(export_dir, "model_int8_full_floatio.tflite"), tfl)

# ---------- Variant E: INT8 full (int8 I/O)  [OPTIONAL] ----------
try:
    print("Converting → INT8 (full, int8 I/O) …")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: rep_dataset()
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    save_bytes(os.path.join(export_dir, "model_int8_full_int8io.tflite"), tfl)
except Exception as e:
    print("  (Skipping int8 I/O variant – converter/ops not supported):", e)

# ---------- Save scalers & LPF taps ----------
print("\nSaving scalers & LPF taps …")
joblib.dump(feat_scaler, os.path.join(export_dir, "feat_scaler.gz"))
joblib.dump(lab_scaler,  os.path.join(export_dir, "label_scaler.gz"))

sos = ss.butter(8, 4000/8000, btype="low", output="sos")
np.save(os.path.join(export_dir, "lpf_sos.npy"), sos)
print("  wrote feat_scaler.gz / label_scaler.gz / lpf_sos.npy")

print("\n✓  Done – files in", export_dir)
for f in sorted(os.listdir(export_dir)):
    print("   ", f)
