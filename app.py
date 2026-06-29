"""
AI Image Detector — Web Server
Auto-reload: detector_config.json & model.safetensors dipantau otomatis.
Perubahan berlaku tanpa restart server.
"""

import os, json, uuid, time, hashlib, logging, threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError

# ─── App ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True   # HTML/CSS langsung update

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = app.logger

MODEL_DIR  = Path("model")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
AI_KEYWORDS   = {"ai", "fake", "generated", "artificial", "synthetic", "gan", "diffusion", "palsu"}
REAL_KEYWORDS = {"real", "genuine", "authentic", "asli", "natural", "original", "human", "nyata"}


# ─── State (mutable, thread-safe) ────────────────────────────────────────────
_lock  = threading.RLock()
_state = {
    "processor":   None,
    "model":       None,
    "device":      None,
    "labels":      [],
    "cfg":         {},
    "version":     "init",   # berubah tiap reload → frontend auto-refresh
    "model_status": "loading",   # 'loading' | 'ready' | 'reloading' | 'error'
    "status_msg":  "",
}


def _bump_version():
    _state["version"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]


# ─── Label utilities ──────────────────────────────────────────────────────────
def _resolve_labels(cfg: dict, model) -> list[str]:
    """Prioritas: detector_config.json → model.id2label → fallback."""
    if "labels" in cfg:
        return cfg["labels"]
    if hasattr(model, "config") and getattr(model.config, "id2label", None):
        return [model.config.id2label[i]
                for i in sorted(model.config.id2label.keys(), key=int)]
    return ["Real", "AI Generated"]


def label_type(label: str) -> str:
    lower = label.lower()
    if any(k in lower for k in AI_KEYWORDS):   return "ai"
    if any(k in lower for k in REAL_KEYWORDS): return "real"
    return "unknown"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Config reload ────────────────────────────────────────────────────────────
def _reload_config():
    """Baca ulang detector_config.json dan update labels. Non-blocking."""
    cfg_path = MODEL_DIR / "detector_config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        with _lock:
            _state["cfg"] = cfg
            if _state["model"] is not None:
                _state["labels"] = _resolve_labels(cfg, _state["model"])
            _bump_version()
        log.info("✅ Config reloaded — labels: %s", _state["labels"])
    except Exception as e:
        log.warning("Config reload gagal: %s", e)


# ─── Model reload ─────────────────────────────────────────────────────────────
def _reload_model():
    """Load/reload model weights. Dipanggil di background thread."""
    with _lock:
        _state["model_status"] = "reloading"
        _state["status_msg"]   = "Model sedang dimuat ulang..."

    log.info("🔄 Memuat ulang model...")
    try:
        processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR))
        model     = AutoModelForImageClassification.from_pretrained(str(MODEL_DIR))
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        with _lock:
            _state["processor"]    = processor
            _state["model"]        = model
            _state["device"]       = device
            _state["labels"]       = _resolve_labels(_state["cfg"], model)
            _state["model_status"] = "ready"
            _state["status_msg"]   = ""
            _bump_version()

        log.info("✅ Model ready on %s — labels: %s", device, _state["labels"])

    except Exception as e:
        with _lock:
            _state["model_status"] = "error"
            _state["status_msg"]   = str(e)
        log.error("❌ Model reload gagal: %s", e)


# ─── File watcher (background thread) ────────────────────────────────────────
def _watcher():
    """
    Pantau perubahan file setiap 3 detik.
    - detector_config.json → reload config + labels (instan)
    - model.safetensors    → reload model di thread terpisah
    """
    targets = {
        MODEL_DIR / "detector_config.json": 0.0,
        MODEL_DIR / "model.safetensors":    0.0,
    }
    # Catat mtime awal
    for p in targets:
        if p.exists():
            targets[p] = p.stat().st_mtime

    log.info("👁  File watcher aktif — memantau %d file", len(targets))

    while True:
        time.sleep(3)
        for path, last_mt in list(targets.items()):
            if not path.exists():
                continue
            try:
                mt = path.stat().st_mtime
                if mt != last_mt:
                    targets[path] = mt
                    log.info("📝 Perubahan terdeteksi: %s", path.name)

                    if path.name == "detector_config.json":
                        _reload_config()

                    elif path.name == "model.safetensors":
                        t = threading.Thread(target=_reload_model, daemon=True)
                        t.start()

            except Exception as e:
                log.warning("Watcher error [%s]: %s", path.name, e)


# ─── Startup ──────────────────────────────────────────────────────────────────
# 1. Load config dulu
try:
    with open(MODEL_DIR / "detector_config.json", encoding="utf-8") as f:
        _state["cfg"] = json.load(f)
except Exception:
    _state["cfg"] = {}

# 2. Load model (di thread terpisah agar server langsung menyala)
threading.Thread(target=_reload_model, daemon=True).start()

# 3. Jalankan watcher
threading.Thread(target=_watcher, daemon=True).start()


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Frontend poll endpoint — digunakan untuk deteksi perubahan."""
    with _lock:
        return jsonify({
            "version":      _state["version"],
            "model_status": _state["model_status"],
            "status_msg":   _state["status_msg"],
            "labels":       _state["labels"],
            "device":       str(_state["device"]) if _state["device"] else None,
        })


@app.route("/predict", methods=["POST"])
def predict():
    with _lock:
        status = _state["model_status"]

    if status != "ready":
        msg = "Model sedang dimuat, harap tunggu..." if status in ("loading", "reloading") \
              else f"Model error: {_state['status_msg']}"
        return jsonify({"error": msg}), 503

    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung."}), 400

    tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
    try:
        file.save(tmp)
        image = Image.open(tmp).convert("RGB")

        with _lock:
            processor = _state["processor"]
            model     = _state["model"]
            device    = _state["device"]
            labels    = _state["labels"]

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().tolist()

        pred_idx = int(np.argmax(probs))
        return jsonify({
            "prediction":      labels[pred_idx],
            "prediction_type": label_type(labels[pred_idx]),
            "confidence":      round(probs[pred_idx] * 100, 2),
            "scores": [
                {"label": lbl, "score": round(p * 100, 2), "type": label_type(lbl)}
                for lbl, p in zip(labels, probs)
            ],
        })

    except UnidentifiedImageError:
        return jsonify({"error": "File bukan gambar yang valid."}), 400
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Gagal memproses: {e}"}), 500
    finally:
        tmp.unlink(missing_ok=True)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  AI Image Detector  —  Auto-Reload Edition")
    print("  URL : http://localhost:5000")
    print("  Auto-reload: config & model dipantau otomatis")
    print("  Ctrl+C untuk berhenti")
    print("=" * 52 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)