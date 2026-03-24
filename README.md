# 🧬 Teknofest Sağlıkta Yapay Zeka - Kanser Tespiti

Bu proje, Teknofest Sağlıkta Yapay Zeka (Kanser Tespiti) yarışması için geliştirilmiş, yüksek doğruluklu bir genomik varyant patojenite tahmin platformudur.

## 🏗️ Proje Yapısı

```text
├── data/
│   ├── raw/               # ClinVar (3.6GB) ve diğer ham genomik veriler
│   └── processed/         # Anonimleştirilmiş ve dengelenmiş eğitim verisi
├── models/                # Eğitilmiş Hybrid Ensemble (XGB+LGBM) modeli
├── reports/               
│   ├── figures/           # Confusion Matrix, ROC ve PR grafik raporları
│   └── final_report.txt   # Model performans metrikleri özeti
├── src/                   # Ana Kaynak Kodlar
│   ├── data_loader.py     # Veri okuma, anonimleştirme ve filtreleme
│   ├── balance_data.py    # Sınıf dengeleme (Undersampling)
│   ├── model.py           # Hibrit Ensemble (XGB+LGBM) mimarisi
│   ├── train.py           # Eğitim, kalibrasyon ve raporlama scripti
│   ├── predict_sample.py  # Model kullanım (inference) örneği
│   └── kontrol.py         # Teknofest uyumluluk kontrol aracı
├── requirements.txt       # Gerekli kütüphaneler (XGBoost, LightGBM, vb.)
└── README.md              # Proje dokümantasyonu
```

## 🚀 Model Başarısı (Hibrit Ensemble)

Kurulan **XGBoost + LightGBM** topluluk modeli, test verisi üzerinde aşağıdaki performansı göstermiştir:

*   **Doğruluk (Accuracy):** %99.20
*   **ROC-AUC Skoru:** 0.9992
*   **Duyarlılık (Recall):** 0.9933
*   **Bölüm 4 (Deney):** 5-Fold Cross-Validation, Isotonic Calibration ve Threshold Optimization (0.74) kullanılmıştır.

## 🛠️ Kurulum ve Kullanım

1.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Veriyi Hazırlayın:**
    `data/raw/variant_summary.txt` dosyası mevcutken sırasıyla:
    ```bash
    python src/data_loader.py
    python src/balance_data.py
    ```

3.  **Modeli Eğitin:**
    ```bash
    python src/train.py
    ```

4.  **Tahmin Yapın:**
    ```bash
    python src/predict_sample.py
    ```

## 📝 Teknofest Uyumluluk
Proje, Teknofest yarışma kurallarına tamamen uygundur:
- ✅ Tüm veriler anonimdir (`Feature_1...9`).
- ✅ Kimlik veya genomik lokasyon bilgisi sızdırılmaz.
- ✅ Sınıf dengesi (50/50) sağlanmıştır.
- ✅ Metrikler bilimsel rapor formatındadır.
