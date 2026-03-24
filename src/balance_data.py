import pandas as pd
import numpy as np

print("Veri dengeleme islemi basliyor...")

df = pd.read_csv('data/processed/teknofest_official_data.csv')

print(f"Onceki dagilim:")
print(df['Target'].value_counts())
print(f"Toplam: {len(df)}\n")

# --- Dengeleme: Undersampling (Cok olan siniftan kisalt) ---
benign = df[df['Target'] == 0]
pathogenic = df[df['Target'] == 1]

min_count = min(len(benign), len(pathogenic))

# Her iki siniftan esit sayida ornek al (random_state ile tekrarlanabilir)
benign_sample = benign.sample(n=min_count, random_state=42)
pathogenic_sample = pathogenic.sample(n=min_count, random_state=42)

# Birlestir ve karistir
df_balanced = pd.concat([benign_sample, pathogenic_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dengelenmis dagilim:")
print(df_balanced['Target'].value_counts())
print(f"Toplam: {len(df_balanced)}\n")

# --- Sutun isimlerini kontrol et ---
feature_cols = [c for c in df_balanced.columns if c.startswith('Feature_')]
print(f"Feature sayisi: {len(feature_cols)}")
print(f"Kolonlar: {list(df_balanced.columns)}")
print(f"Bos deger: {df_balanced.isnull().sum().sum()}")

# --- Kaydet ---
import os
output_path = 'data/processed/teknofest_balanced.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\nKaydedildi: {output_path}")

import os
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"Dosya boyutu: {size_mb:.2f} MB")
