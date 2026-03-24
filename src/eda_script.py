import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

file_path = 'data/variant_summary.txt'
try:
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"=== VERI SETI BILGISI ===")
    print(f"Dosya Boyutu: {file_size_gb:.2f} GB\n")

    # Dosya çok büyük olduğu için ilk 10.000 satırı okuyarak genel bir analiz çıkarıyoruz
    # clinvar variant_summary.txt genellikle tab ayrılmış (TSV) olur
    df = pd.read_csv(file_path, sep='\t', nrows=10000, low_memory=False)

    print(f"Toplam Sütun Sayısı: {df.shape[1]}")
    print(f"İlk 10.000 Satır için Özet Analiz:\n")
    
    # Eksik veri oranları
    eksik_yuzde = (df.isnull().sum() / len(df)) * 100
    eksik_df = pd.DataFrame({'Eksik_Sayisi': df.isnull().sum(), 'Yuzde_%': eksik_yuzde})
    eksik_df = eksik_df[eksik_df['Eksik_Sayisi'] > 0].sort_values(by='Yuzde_%', ascending=False)
    
    print("Sütun Başlıklı Eksik Veri Oranları (İlk 10,000 satır için):")
    print(eksik_df.to_string())
    
    print("\nVeri Tipleri Özeti:")
    print(df.dtypes.value_counts().to_string())
    
except Exception as e:
    print(f"Hata oluştu: {str(e)}")
