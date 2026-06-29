# 🤖 AI Image Detector

Web server untuk mendeteksi apakah sebuah gambar dibuat oleh AI atau merupakan gambar asli, menggunakan model fine-tuned berbasis HuggingFace Transformers.

---

## 📁 Struktur Folder

```
ai-detector/
├── model/
│   ├── model.safetensors     ← model weights (dari Google Drive)
│   ├── config.json           ← konfigurasi arsitektur model (HuggingFace)
│   └── detector_config.json  ← label & threshold kustom
│
├── templates/
│   └── index.html            ← UI web (drag & drop)
│
├── uploads/                  ← folder sementara (auto-dibersihkan)
│
├── app.py                    ← Flask web server
├── requirements.txt          ← daftar dependensi Python
├── run_server.bat            ← launcher Windows
└── README.md
```

---

## ⚙️ Setup

### 1. Letakkan file model
Download file dari Google Drive dan taruh di folder `model/`:
- `model.safetensors`
- `config.json`
- `detector_config.json`

### 2. Edit `detector_config.json`
Sesuaikan label dengan hasil training kamu:
```json
{
  "labels": ["Real", "AI Generated"],
  "threshold": 0.5
}
```
> **Penting:** urutan label harus sesuai dengan `id2label` di `config.json`  
> Contoh: jika `id2label = {"0": "REAL", "1": "FAKE"}` → labels = `["REAL", "FAKE"]`

### 3. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
```

### 4. Install dependensi
```bash
pip install -r requirements.txt
```
> Untuk CPU saja, ganti torch dengan versi CPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 🚀 Menjalankan Server

**Windows:**
```
Klik dua kali run_server.bat
```

**Manual (semua OS):**
```bash
python app.py
```

Buka browser di: **http://localhost:5000**

---

## 🌐 API Endpoint

| Method | Endpoint   | Deskripsi                          |
|--------|------------|------------------------------------|
| GET    | `/`        | Halaman web utama                  |
| GET    | `/health`  | Status server & device             |
| POST   | `/predict` | Prediksi gambar (form-data: image) |

### Contoh response `/predict`:
```json
{
  "prediction": "AI Generated",
  "prediction_type": "ai",
  "confidence": 94.73,
  "scores": [
    {"label": "Real",         "score": 5.27,  "type": "real"},
    {"label": "AI Generated", "score": 94.73, "type": "ai"}
  ]
}
```

---

## 🔧 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `Cannot load model` | Pastikan `config.json` ada dan valid (HuggingFace format) |
| `KeyError label` | Cek urutan label di `detector_config.json` |
| `CUDA out of memory` | Tidak ada GPU, model otomatis pakai CPU |
| File tidak ditemukan | Pastikan `model/` berisi ketiga file model |
