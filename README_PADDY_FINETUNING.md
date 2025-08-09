# CvT Fine-tuning untuk Klasifikasi Penyakit Tanaman Padi

Panduan lengkap untuk melakukan fine-tuning model CvT-21 untuk klasifikasi penyakit tanaman padi dengan 10 kelas. Repository ini telah dikonfigurasi untuk mendukung **train-validation-test split** yang terpisah.

## Setup di Google Colab

### 1. Persiapan Dataset

Struktur dataset harus dalam format ImageFolder dengan **3 direktori terpisah**:

```
CvT/
├── paddy_disease_dataset/
│   ├── train/ (80% data - untuk training)
│   │   ├── bacterial_leaf_blight/
│   │   ├── bacterial_leaf_streak/
│   │   ├── bacterial_panicle_blight/
│   │   ├── blast/
│   │   ├── brown_spot/
│   │   ├── dead_heart/
│   │   ├── downy_mildew/
│   │   ├── hispa/
│   │   ├── normal/
│   │   └── tungro/
│   ├── val/ (10% data - untuk validation selama training)
│   │   ├── [same 10 classes]
│   └── test/ (10% data - untuk final evaluation)
│       ├── [same 10 classes]
└── CvT-21-224x224-IN-1k.pth
```

```
CvT/
├── paddy_disease_dataset/
│   ├── train/
│   │   ├── bacterial_leaf_blight/
│   │   ├── bacterial_leaf_streak/
│   │   ├── bacterial_panicle_blight/
│   │   ├── blast/
│   │   ├── brown_spot/
│   │   ├── dead_heart/
│   │   ├── downy_mildew/
│   │   ├── hispa/
│   │   ├── normal/
│   │   └── tungro/
│   └── val/
│       ├── bacterial_leaf_blight/
│       ├── bacterial_leaf_streak/
│       ├── bacterial_panicle_blight/
│       ├── blast/
│       ├── brown_spot/
│       ├── dead_heart/
│       ├── downy_mildew/
│       ├── hispa/
│       ├── normal/
│       └── tungro/
└── CvT-21-224x224-IN-1k.pth
```

### 2. Persiapan File

**Download dan setup manual**:

1. **Dataset**: Download dataset Anda dan extract ke `/content/CvT/paddy_disease_dataset/`
   - **Pastikan ada 3 folder: `train/`, `val/`, `test/`**
   - **Rasio: Train 80% : Val 10% : Test 10%**
2. **Pretrained weights**: Download `CvT-21-224x224-IN-1k.pth` dan letakkan di `/content/CvT/`

**Cara setup**:

- Use file browser di Colab untuk upload/copy files
- Atau download langsung menggunakan `wget` atau `gdown` di Colab
- Direktori `paddy_disease_dataset/` sudah ada dalam repository (kosong)

### 3. Jalankan Training

1. Buka `run_training_colab.ipynb` di Google Colab
2. Download dan letakkan dataset + weights di path yang benar
3. Jalankan semua cell secara berurutan
4. Notebook akan memverifikasi file dan memulai training

## Konfigurasi Training

File konfigurasi: `experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml` (dikonfigurasi khusus untuk paddy dataset)

### Parameter Utama:

- **Jumlah kelas**: 10 (9 penyakit + 1 normal)
- **Batch size**: 32
- **Image size**: 224x224
- **Epochs**: 30
- **Learning rate**: 5e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing

### Augmentasi Data:

- Random resize crop
- Color jitter
- Horizontal flip
- Normalization (ImageNet stats)

## Monitoring Training

### 1. Real-time Monitoring

```python
# Di cell terpisah, jalankan untuk melihat progress
!tail -f /content/output/log.txt
```

### 2. TensorBoard (opsional)

```python
%load_ext tensorboard
%tensorboard --logdir /content/output
```

## Output Training

Setelah training selesai, file berikut akan tersedia di `/content/output/`:

- `best.pth`: Model dengan validation accuracy terbaik
- `latest.pth`: Model dari epoch terakhir
- `log.txt`: Log lengkap training
- `events.out.tfevents.*`: File TensorBoard

## Evaluasi Model

### 1. Evaluasi pada Validation Set (selama training)

Training otomatis mengevaluasi pada validation set setiap epoch. Hasil terbaik disimpan sebagai `best.pth`.

### 2. Evaluasi pada Test Set (untuk hasil final)

Setelah training selesai, evaluasi model pada test set yang belum pernah dilihat:

```bash
# Evaluasi menggunakan test set
python tools/final_test.py \
    --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml \
    --model-file /content/output/best.pth \
    --dataset-type test

# Atau bandingkan dengan validation set
python tools/final_test.py \
    --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml \
    --model-file /content/output/best.pth \
    --dataset-type val
```

**Catatan Penting:**

- Test set hanya digunakan untuk evaluasi final, bukan untuk tuning hyperparameter
- Validation set digunakan untuk monitoring training dan early stopping
- Train set (80%) untuk training, val set (10%) untuk validation, test set (10%) untuk evaluasi final

### 3. Inference pada Gambar Baru

Untuk inference pada gambar tunggal, Anda perlu memodifikasi `tools/test.py` atau membuat script terpisah.

## Tips Optimisasi

### 1. Jika Memory Error:

- Kurangi batch size dari 32 ke 16 atau 8
- Edit `TRAIN.BATCH_SIZE_PER_GPU` di file `experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml`

### 2. Jika Training Lambat:

- Gunakan GPU T4 atau lebih tinggi di Colab Pro
- Kurangi resolusi image dari 224 ke 192

### 3. Jika Overfitting:

- Tingkatkan `LOSS.LABEL_SMOOTHING` dari 0.1 ke 0.2
- Tambahkan dropout dengan mengedit `MODEL.SPEC.DROP_RATE`

## Troubleshooting

### Error: "CUDA out of memory"

```yaml
# Di experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml, ubah:
TRAIN:
  BATCH_SIZE_PER_GPU: 16 # atau 8
```

### Error: "FileNotFoundError" untuk dataset

Pastikan dataset sudah diupload dengan benar:

```bash
# Cek apakah dataset ada
!ls /content/paddy_disease_dataset/  # atau
!ls /content/CvT/paddy_disease_dataset/
```

### Error: "Model checkpoint not found"

Pastikan file pretrained weights sudah diupload:

```bash
# Cek apakah weights ada
!ls /content/CvT-21-224x224-IN-1k.pth  # atau
!ls /content/CvT/CvT-21-224x224-IN-1k.pth
```

Pastikan file pretrained weights ada:

```yaml
MODEL:
  PRETRAINED: "/content/CvT/CvT-21-224x224-IN-1k.pth"
```

## Backup Hasil

Hasil training akan otomatis dibuat dalam format archive yang bisa didownload:

- File archive: `/content/cvt_paddy_results.zip`
- Download melalui file browser Colab atau widget download di notebook

## Kustomisasi

### 1. Mengganti Jumlah Kelas

Edit di `experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml`:

```yaml
MODEL:
  NUM_CLASSES: 10 # Ubah sesuai jumlah kelas Anda
```

### 2. Mengganti Arsitektur Model

Untuk menggunakan CvT-13 atau CvT-W24:

```yaml
MODEL:
  PRETRAINED: "/path/to/CvT-13-224x224-IN-1k.pth" # atau CvT-W24
  SPEC:
    # Sesuaikan parameter arsitektur
```

### 3. Transfer Learning vs Fine-tuning

Untuk freeze backbone dan hanya train classifier:

```yaml
FINETUNE:
  FROZEN_LAYERS: ["backbone"] # Freeze semua layer kecuali head
```

## Kontribusi

Jika menemukan bug atau ingin berkontribusi, silakan buat issue atau pull request di repository ini.

## Lisensi

Mengikuti lisensi dari repository Microsoft CvT original.
