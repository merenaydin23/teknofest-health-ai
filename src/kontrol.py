import pandas as pd

df = pd.read_csv('data/processed/teknofest_balanced.csv')
cols = list(df.columns)
feature_cols = [c for c in cols if c.startswith('Feature_')]
has_id = any(c in cols for c in ['CHROM', 'POS', 'REF', 'ALT', 'GeneSymbol', 'Name'])
ratio = df['Target'].mean()
missing = df.isnull().sum().sum()
non_num = df.drop(columns=['Target']).select_dtypes(exclude=['number']).columns.tolist()

results = {
    'Kimlik kolonu yok (Anonim)': not has_id,
    'Target sadece 0 ve 1 iceriyor': set(df['Target'].unique()) == {0, 1},
    'Sinif dengesi 50/50': 0.45 <= ratio <= 0.55,
    'Bos veri yok': missing == 0,
    'Tum feature kolonlari numerik': len(non_num) == 0,
    'Satir sayisi 50.000+': len(df) >= 50000,
    'Feature sayisi uygun (5-100 arasi)': 5 <= len(feature_cols) <= 100
}

print("=" * 55)
print("  TEKNOFEST SON UYUMLULUK KONTROL RAPORU")
print("=" * 55)
for k, v in results.items():
    durum = "OK" if v else "FAIL"
    print(f"  [{durum}] {k}")

gecti = sum(results.values())
toplam = len(results)
print("=" * 55)
print(f"  SONUC: {gecti}/{toplam} kontrol gecti")

if gecti == toplam:
    print("  DURUM: TEKNOFEST FORMATINA TAMAMEN UYUMLU!")
else:
    print("  DURUM: Bazi kontroller basarisiz!")

print("=" * 55)
print(f"  Toplam Satir : {len(df):,}")
print(f"  Feature Sayisi: {len(feature_cols)}")
print(f"  Pathogenic   : {df['Target'].sum():,} (%{round(ratio*100,1)})")
print(f"  Benign       : {(df['Target']==0).sum():,} (%{round((1-ratio)*100,1)})")
print(f"  Bos Deger    : {missing}")
print("=" * 55)
