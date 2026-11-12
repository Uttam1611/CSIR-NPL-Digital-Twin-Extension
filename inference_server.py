import os
import json
import pickle
import traceback
import uuid
import logging
import numpy as np
from scipy.signal import stft, hilbert, butter, filtfilt, resample
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# ── Task 1 / Task 2 extension: dataset generation + measurement uncertainty ──
# These two packages are new, self-contained additions living next to this
# file (uncertainty/, storage/). Nothing above this point, and nothing in the
# existing feature-extraction / model-loading code below, was changed.
from uncertainty import UncertaintyEngine
from storage.config import load_postgres_config
from storage.connection import PostgresConnectionManager
from storage.schema import ensure_schema
from storage.repository import InferenceCycleRepository
from storage.models import InferenceCycleRecord

logger = logging.getLogger("digital_twin.inference_server")

# ── Load artifacts ─────────────────────────────────────────────
ARTIFACTS_DIR = "./dt_model_artifacts"
META_PATH     = os.path.join(ARTIFACTS_DIR, "meta_learner.pkl")
CONFIG_PATH   = os.path.join(ARTIFACTS_DIR, "feature_config.json")

ONNX_PATHS = {
    "cnn"        : os.path.join(ARTIFACTS_DIR, "cnn_model.onnx"),
    "lstm"       : os.path.join(ARTIFACTS_DIR, "lstm_model.onnx"),
    "transformer": os.path.join(ARTIFACTS_DIR, "transformer_model.onnx"),
    "hybrid"     : os.path.join(ARTIFACTS_DIR, "hybrid_model.onnx"),
}

print("[INFO] Loading model artifacts...")

try:
    import onnxruntime as ort
    sessions = {}
    for name, path in ONNX_PATHS.items():
        sessions[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"[SUCCESS] Loaded {name} model.")
except Exception as e:
    print(f"[FATAL] Could not load ONNX models: {e}")
    raise

try:
    with open(META_PATH, "rb") as f:
        meta_learner = pickle.load(f)
    print(f"[SUCCESS] Meta-learner loaded. Expects {meta_learner.n_features_in_} features.")
except Exception as e:
    print(f"[FATAL] Could not load meta-learner: {e}")
    raise

try:
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    print(f"[SUCCESS] Feature config loaded.")
except Exception as e:
    print(f"[WARNING] feature_config.json not found, using defaults: {e}")
    cfg = {}

# ── Config ─────────────────────────────────────────────────────
SEGMENT_LENGTH = cfg.get("segment_length", 2048)
TARGET_SR      = cfg.get("training_sr",    12000)
INFERENCE_SR   = cfg.get("inference_sr",   1000)
NPERSEG        = cfg.get("nperseg",        512)
NOVERLAP       = cfg.get("noverlap",       384)
FEATURE_SHAPE  = tuple(cfg.get("feature_shape", [17, 769]))
CLASS_NAMES    = cfg.get("class_names", ["Ball", "Normal", "OuterRace", "InnerRace"])

print(f"[INFO] SR: {TARGET_SR}Hz | Segment: {SEGMENT_LENGTH} | Features: {FEATURE_SHAPE}")

# ── Uncertainty engine (Task 2) ─────────────────────────────────
# Independent of MQTT/dashboard/DB by design -- see uncertainty/__init__.py.
uncertainty_engine = UncertaintyEngine(
    window_size=int(os.environ.get("DT_TYPE_A_WINDOW", "30")),
)

# ── PostgreSQL dataset-generation pipeline (Task 1) ─────────────
# Fails soft: if Postgres is unreachable at startup, the server still starts
# and /predict still works -- it just logs a warning and skips DB writes,
# rather than the dataset-logging feature being able to take down inference.
STORE_FEATURE_MATRIX = os.environ.get("DT_STORE_FEATURE_MATRIX", "false").lower() == "true"
cycle_repository: Optional[InferenceCycleRepository] = None
try:
    pg_config = load_postgres_config()
    pg_manager = PostgresConnectionManager(pg_config)
    ensure_schema(pg_manager)
    cycle_repository = InferenceCycleRepository(pg_manager, store_feature_matrix=STORE_FEATURE_MATRIX)
    print(f"[SUCCESS] PostgreSQL dataset-generation pipeline ready ({pg_config.host}:{pg_config.port}/{pg_config.database}).")
except Exception as e:
    print(f"[WARNING] PostgreSQL unavailable at startup -- dataset logging disabled until it recovers: {e}")
    cycle_repository = None


def _feature_summary(features: np.ndarray) -> Dict[str, Any]:
    """
    Lightweight, storable summary of the (17, 769) STFT+envelope feature
    matrix -- mean/std/min/max per row-block isn't meaningful here, so this
    just captures whole-matrix summary stats. This is what populates the
    `preprocessed_features` column by default; the full matrix itself is only
    stored (gzip-compressed, in `feature_matrix_bytes`) if DT_STORE_FEATURE_MATRIX=true,
    since it's ~52 KB/row uncompressed and this pipeline is meant to log every cycle.
    """
    return {
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
    }


def _persist_cycle(record: InferenceCycleRecord) -> None:
    """Runs in a FastAPI BackgroundTask, i.e. after the HTTP response has
    already been sent -- a slow or failing DB write never adds latency to
    the /predict response the dashboard/control loop is waiting on."""
    if cycle_repository is None:
        return
    ok = cycle_repository.insert(record)
    if not ok:
        logger.warning("Inference cycle %s was not persisted (see storage logs above).", record.cycle_uuid)


# ── Feature extraction (identical to training Cell 6) ──────────
def extract_features(seg, nperseg=NPERSEG, noverlap=NOVERLAP, fs=TARGET_SR):
    b, a = butter(4, 50 / (fs / 2), 'high', analog=False)
    try:
        seg_filt = filtfilt(b, a, seg)
    except Exception:
        seg_filt = seg

    _, _, Z  = stft(seg, fs=fs, nperseg=nperseg, noverlap=noverlap)
    stft_mag = np.log1p(np.abs(Z).T).astype(np.float32)
    stft_mag = (stft_mag - stft_mag.mean()) / (stft_mag.std() + 1e-10)
    T        = stft_mag.shape[0]

    env      = np.abs(hilbert(seg_filt))
    env_down = env[::32]
    env_spec = np.abs(np.fft.fft(env_down))[:len(env_down) // 2]

    def align(arr, T):
        if len(arr) < T:
            return np.pad(arr, (0, T - len(arr)), 'edge')
        return arr[:T]

    env_norm = align(env_down, T)
    env_norm = (env_norm - env_norm.mean()) / (env_norm.std() + 1e-10)

    env_spec_interp = np.interp(
        np.linspace(0, len(env_spec) - 1, T),
        np.arange(len(env_spec)), env_spec
    )
    env_spec_norm = (
        (env_spec_interp - env_spec_interp.mean()) /
        (env_spec_interp.std() + 1e-10)
    )
    F             = stft_mag.shape[1]
    env_feat      = np.tile(env_norm[:, None],      (1, F)).astype(np.float32)
    env_spec_feat = np.tile(env_spec_norm[:, None], (1, F)).astype(np.float32)
    combined      = np.concatenate([stft_mag, env_feat, env_spec_feat], axis=1)

    H, W = FEATURE_SHAPE  # (17, 769)

    # Fix rows
    if combined.shape[0] < H:
        combined = np.pad(combined, ((0, H - combined.shape[0]), (0, 0)))
    elif combined.shape[0] > H:
        combined = combined[:H, :]

    # Fix columns
    if combined.shape[1] < W:
        combined = np.pad(combined, ((0, 0), (0, W - combined.shape[1])))
    elif combined.shape[1] > W:
        combined = combined[:, :W]

    print(f"[DEBUG] extract_features output shape: {combined.shape}")  # must be (17, 769)

    return combined.astype(np.float32)

def preprocess_window(raw_window: List[float]) -> np.ndarray:

    sig = np.array(raw_window, dtype=np.float32)

    target_len = int(len(sig) * TARGET_SR / INFERENCE_SR)

    sig_resampled = resample(sig, target_len).astype(np.float32)

    if len(sig_resampled) >= SEGMENT_LENGTH:
        seg = sig_resampled[:SEGMENT_LENGTH]
    else:
        seg = np.pad(
            sig_resampled,
            (0, SEGMENT_LENGTH - len(sig_resampled)),
            'edge'
        )

    features = extract_features(seg)

    print(f"[DEBUG] Extracted feature shape: {features.shape}")  # (17, 769)

    # Change this line in preprocess_window:
    features = features[np.newaxis, :, :]      # (1, 17, 769) — just ONE newaxis

    print(f"[DEBUG] Final input shape: {features.shape}")  # should be (1, 17, 769)

    return features

def run_inference(features: np.ndarray):
    print("\n========== RUN_INFERENCE ==========")
    print(f"[DEBUG] Incoming feature shape: {features.shape}")  # (1, 17, 769)

    all_probs = []

    for name in ["cnn", "lstm", "transformer", "hybrid"]:
        sess = sessions[name]
        try:
            inp_name = sess.get_inputs()[0].name
            expected = sess.get_inputs()[0].shape
            print(f"\n[MODEL] {name} expects: {expected}")
            print(f"[MODEL] {name} sending shape: {features.shape}")

            out = sess.run(None, {inp_name: features})[0][0]
            print(f"[MODEL] {name} output: {out}")
            all_probs.append(out)

        except Exception as e:
            print(f"\n[MODEL ERROR] {name}: {e}")
            raise e

    print(f"[META] Individual output lengths: {[len(p) for p in all_probs]}")
    stacked   = np.concatenate(all_probs).reshape(1, -1)
    print(f"[META] Stacked shape: {stacked.shape}")  # must be (1, 16)

    meta_pred  = meta_learner.predict(stacked)[0]
    meta_proba = meta_learner.predict_proba(stacked)[0]

    fault_class = CLASS_NAMES[int(meta_pred)]
    confidence  = float(np.max(meta_proba))

    all_probs_d = {
        CLASS_NAMES[i]: round(float(meta_proba[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    print(f"[FINAL] {fault_class} ({confidence:.3f})")
    print("===================================\n")

    return fault_class, confidence, all_probs_d
# ── FastAPI ────────────────────────────────────────────────────
app = FastAPI(title="Digital Twin Inference Server")


class PredictRequest(BaseModel):
    window: List[float]

    # Everything below is NEW and OPTIONAL, defaulting to None. The existing
    # caller behavior (sending only `window`) is unchanged -- this codebase's
    # /predict endpoint previously had no way to receive rpm/current/voltage/
    # frequency/temperature at all, and Task 1 requires logging them per
    # cycle, so these fields are added here as the natural place for a caller
    # (e.g. control_panel.py, once it is wired up, or the mock-sensors
    # reference pipeline) to attach them when available.
    rpm:            Optional[float] = None
    current:        Optional[float] = None
    voltage:        Optional[float] = None
    frequency:      Optional[float] = None
    temperature:    Optional[float] = None
    extra_sensors:  Optional[Dict[str, Any]] = None  # any sensor not given its own field above


class UncertaintyChannel(BaseModel):
    type_a_standard_uncertainty: float
    type_b_standard_uncertainty: float
    combined_standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float
    n_samples: int


class PredictResponse(BaseModel):
    fault_class:   str
    confidence:    float
    probabilities: dict
    window_size:   int
    resampled_sr:  int
    uncertainty:   Optional[Dict[str, UncertaintyChannel]] = None  # per-channel GUM breakdown
    cycle_id:      Optional[str] = None  # inference_cycles.cycle_uuid, if the row was persisted


@app.get("/health")
def health():
    return {
        "status"          : "ok",
        "model"           : "CNN + LSTM + Transformer + Hybrid → Stacking (95.1%)",
        "classes"         : CLASS_NAMES,
        "inference_sr"    : INFERENCE_SR,
        "target_sr"       : TARGET_SR,
        "dataset_logging" : "enabled" if cycle_repository is not None else "disabled (PostgreSQL unreachable at startup)",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks):

    print("\n================ /predict CALLED ================")

    try:

        print(f"[DEBUG] Window length: {len(req.window)}")

        features = preprocess_window(req.window)

        print(f"[DEBUG] Features shape: {features.shape}")

        fault_class, confidence, all_probs = run_inference(features)

        print(f"[DEBUG] Final prediction: {fault_class}")
        print("=================================================\n")

        # ── Task 2: measurement uncertainty (separate from `confidence` above,
        # which is model confidence, not measurement uncertainty) ──
        vibration_rms = float(np.sqrt(np.mean(np.square(req.window)))) if req.window else None
        readings = {"vibration": vibration_rms}
        if req.current is not None:
            readings["current"] = req.current
        if req.voltage is not None:
            readings["voltage"] = req.voltage
        if req.frequency is not None:
            readings["frequency"] = req.frequency
        if req.temperature is not None:
            readings["temperature"] = req.temperature

        uncertainty_results = uncertainty_engine.evaluate(readings)
        uncertainty_dict = UncertaintyEngine.as_dict(uncertainty_results)
        vibration_uncertainty = uncertainty_results.get("vibration")

        # ── Task 1: log this inference cycle to PostgreSQL (fire-and-forget,
        # after the response is sent -- see _persist_cycle docstring) ──
        record = InferenceCycleRecord(
            rpm=req.rpm,
            current_a=req.current,
            voltage_v=req.voltage,
            frequency_hz=req.frequency,
            temperature_c=req.temperature,
            vibration_rms_g=vibration_rms,
            fault_class=fault_class,
            confidence=confidence,
            model_probabilities=all_probs,
            type_a_uncertainty=vibration_uncertainty.type_a_standard_uncertainty if vibration_uncertainty else None,
            type_b_uncertainty=vibration_uncertainty.type_b_standard_uncertainty if vibration_uncertainty else None,
            combined_uncertainty=vibration_uncertainty.combined_standard_uncertainty if vibration_uncertainty else None,
            uncertainty_detail=uncertainty_dict,
            extra_sensors=req.extra_sensors,
            preprocessed_features=_feature_summary(features),
            feature_matrix_shape=list(features.shape),
            feature_matrix_bytes=features.astype(np.float32).tobytes() if STORE_FEATURE_MATRIX else None,
            inference_sr_hz=INFERENCE_SR,
            target_sr_hz=TARGET_SR,
            window_size=SEGMENT_LENGTH,
        )
        if cycle_repository is not None:
            background_tasks.add_task(_persist_cycle, record)

        return PredictResponse(
            fault_class=fault_class,
            confidence=confidence,
            probabilities=all_probs,
            window_size=SEGMENT_LENGTH,
            resampled_sr=TARGET_SR,
            uncertainty=uncertainty_dict,
            cycle_id=str(record.cycle_uuid),
        )

    except Exception as e:

        print("\n!!!!!!!!!! INFERENCE ERROR !!!!!!!!!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    print("\n=============================================")
    print("   DIGITAL TWIN INFERENCE SERVER")
    print("=============================================\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )