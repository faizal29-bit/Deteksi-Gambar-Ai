## Deteksi Gambar AI Menggunakan Model SMOGY (Swin Transformer)
Deskripsi

Proyek ini merupakan implementasi model Deep Learning untuk mendeteksi apakah suatu gambar merupakan gambar asli (Human) atau gambar hasil Artificial Intelligence (AI-Generated).

Model yang digunakan adalah SMOGY AI Images Detector berbasis Swin Transformer yang kemudian dilakukan fine-tuning menggunakan dataset AI vs Human Generated Dataset dari Kaggle.

## Dataset

Dataset yang digunakan berasal dari Kaggle:

AI vs Human Generated Dataset

https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

Dataset terdiri dari dua kelas, yaitu:

HUMAN → Gambar asli (Real Image)
AI → Gambar hasil Artificial Intelligence

Struktur dataset:

dataset/
├── AI/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── HUMAN/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

## Model yang Digunakan

Model yang digunakan adalah model SMOGY AI Images Detector dari Hugging Face.

https://huggingface.co/Smogy/SMOGY-Ai-images-detector

Model ini dibangun menggunakan arsitektur Swin Transformer (Shifted Window Transformer) yang telah melalui proses pretrained sehingga memiliki kemampuan awal dalam mengenali karakteristik gambar AI maupun gambar asli.

Pada program ini diterapkan metode Transfer Learning, yaitu:

Layer backbone Swin Transformer dibekukan (freeze)
Hanya layer classifier yang dilatih kembali (fine-tuning)

Pendekatan ini membuat proses pelatihan lebih cepat, membutuhkan data yang lebih sedikit, serta mampu mempertahankan performa model.

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

# folder fine-tunning model adalah program python yang digunakan untuk training model SMOGY menggunakan dataset alesandra