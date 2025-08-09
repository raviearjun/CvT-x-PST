# Paddy Disease Dataset

Dataset direktori ini akan diisi dengan dataset klasifikasi penyakit tanaman padi.

## Struktur yang diharapkan:

```
paddy_disease_dataset/
├── train/
│   ├── bacterial_leaf_blight/
│   ├── bacterial_leaf_streak/
│   ├── bacterial_panicle_blight/
│   ├── blast/
│   ├── brown_spot/
│   ├── dead_heart/
│   ├── downy_mildew/
│   ├── hispa/
│   ├── normal/
│   └── tungro/
├── val/
│   ├── bacterial_leaf_blight/
│   ├── bacterial_leaf_streak/
│   ├── bacterial_panicle_blight/
│   ├── blast/
│   ├── brown_spot/
│   ├── dead_heart/
│   ├── downy_mildew/
│   ├── hispa/
│   ├── normal/
│   └── tungro/
└── test/
    ├── bacterial_leaf_blight/
    ├── bacterial_leaf_streak/
    ├── bacterial_panicle_blight/
    ├── blast/
    ├── brown_spot/
    ├── dead_heart/
    ├── downy_mildew/
    ├── hispa/
    ├── normal/
    └── tungro/
```

## Catatan:

- Direktori ini kosong di repository dan tidak diikutkan dalam git (gitignore)
- Di Google Colab, download dan isi direktori ini dengan dataset Anda
- Total 10 kelas: 9 penyakit + 1 normal
- Format gambar: jpg, jpeg, png
- **Rasio dataset: Train 80% : Val 10% : Test 10%**
