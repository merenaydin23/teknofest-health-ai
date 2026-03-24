# Teknofest Health AI - Kanser Tespiti

Bu proje, Teknofest Sağlıkta Yapay Zeka (Kanser Tespiti) yarışması için hazırlanmış bir makine öğrenmesi/derin öğrenme proje taslağıdır.

## Proje Yapısı

\`\`\`
├── data/
│   ├── raw/               # İşlenmemiş, orijinal veriler (görüntüler veya etiketler)
│   └── processed/         # Model eğitimi için hazır hale getirilmiş veriler
├── models/                # Eğitilmiş ve kaydedilmiş model dosyaları (.pth, .h5, .pkl vb.)
├── notebooks/             # Exploratory Data Analysis (EDA) ve deneme not defterleri
├── src/                   # Projenin ana kaynak kodları
│   ├── data_loader.py     # Veri okuma ve önişleme
│   ├── model.py           # Model mimarisi
│   ├── train.py           # Model eğitim scripti
│   ├── evaluate.py        # Model metriklerini ölçme scripti
│   └── utils.py           # Yardımcı fonksiyonlar
├── requirements.txt       # Gerekli kütüphaneler
└── README.md              # Proje dökümantasyonu
\`\`\`

## Kurulum ve Kullanım

1. **Bağımlılıkları Yükleyin:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Veriyi Ekleyin:**
   Verisetini \`data/raw/\` klasörüne yerleştirin.

3. **Eğitim:**
   Modeli eğitmek için terminal veya komut istemisinde \`src\` dizini altındaki scriptleri çalıştırın.
   \`\`\`bash
   python src/train.py
   \`\`\`

Bu altyapı veri temizleme ve model eğitimi işlemleri için hazır durumdadır. Yarışma boyunca geliştirerek devam edeceğiz.
