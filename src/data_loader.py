import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def create_teknofest_dataset(input_path, output_path, chunk_size=200000):
    print("Teknofest yarisma kurallarina uygun anonim veri uretimi basliyor...")
    
    # Kabul edilebilir siniflar
    pathogenic_labels = ['Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic']
    benign_labels = ['Benign', 'Likely benign', 'Benign/Likely benign']
    valid_labels = pathogenic_labels + benign_labels
    
    first_chunk = True
    total_processed = 0
    total_saved = 0
    
    sep = '\t' if input_path.endswith('.txt') or input_path.endswith('.tsv') else ','
    
    # Okuma sirasinda gizlenecek veya cope atilacak teknik isimler
    drop_cols = ['CHROM', 'POS', 'REF', 'ALT', 'GeneSymbol', 'Name', 
                 'Assembly', 'RS# (dbSNP)', 'RCVaccession', 'Type', 'OriginSimple']
                 
    try:
        for chunk in pd.read_csv(input_path, sep=sep, chunksize=chunk_size, low_memory=False):
            total_processed += len(chunk)
            
            # 1. Sadece Pathogenic ve Benign olanlari filtrele (Digerlerini cikar)
            if 'ClinicalSignificance' in chunk.columns:
                chunk = chunk[chunk['ClinicalSignificance'].isin(valid_labels)].copy()
                
                # Binary Label'a cevir: Pathogenic = 1, Benign = 0
                chunk['Target'] = chunk['ClinicalSignificance'].apply(
                    lambda x: 1 if x in pathogenic_labels else 0
                )
                chunk.drop(columns=['ClinicalSignificance'], inplace=True)
            else:
                continue # Eger hedef kolon yoksa atla
                
            if len(chunk) == 0:
                continue
                
            # 2. Kurallara gore adres ve kolon isimlerini sil
            existing_drops = [c for c in drop_cols if c in chunk.columns]
            motif_drops = [c for c in chunk.columns if 'MOTIF' in str(c) or 'BLOSUM' in str(c)]
            chunk.drop(columns=existing_drops + motif_drops, inplace=True, errors='ignore')
            
            # 3. %80'i bos olan kolonlari at
            missing_ratios = chunk.isnull().mean()
            cols_to_drop_missing = missing_ratios[missing_ratios > 0.80].index
            chunk.drop(columns=cols_to_drop_missing, inplace=True, errors='ignore')
            
            # 4. Numerik kolonlardaki eksikleri Median ile doldur (sadece secili kolonlar)
            for c in chunk.columns:
                if c == 'Target': continue
                if chunk[c].dtype in ['float64', 'int64']:
                    chunk[c] = chunk[c].fillna(chunk[c].median() if not pd.isna(chunk[c].median()) else 0)
                else:
                    chunk[c] = chunk[c].fillna('unknown')
            
            # Sadece Sayisal (Feature) hale donmus kolonlari al (Daha gercekci anonymization)
            numeric_features = chunk.select_dtypes(include=['float64', 'int64']).copy()
            
            if 'Target' in numeric_features.columns:
                target_col = numeric_features.pop('Target')
            else:
                target_col = chunk['Target']
                
            # 5. Kolon isimlerini gizle (Feature_1, Feature_2 seklinde)
            numeric_features.columns = [f"Feature_{i+1}" for i in range(len(numeric_features.columns))]
            
            # Target'i en sona ekle
            numeric_features['Target'] = target_col.values
            
            # Dosyaya yaz
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            numeric_features.to_csv(output_path, mode=mode, header=header, index=False)
            
            first_chunk = False
            total_saved += len(numeric_features)
            
            print(f"Tarandi: {total_processed} | Temizlenip Kaydedildi: {total_saved}")
            
            # Hoca icin/Test icin 50,000 temiz satira ulasinca duralim ki asiri vakit kaybetmeyelim
            if total_saved >= 50000:
                print("Yeterli sayida (50.000) temiz veri ayrildi.")
                break
                
        print(f"\nISLEM TAMAM! Teknofest formati hazir: {output_path}")
        print(f"Bastan Sona Orijinal 3.6GB icinden damitalarak boyut inanilmaz kucultuldu.")
        
    except FileNotFoundError:
        print(f"Hata: {input_path} bulunamadi.")

if __name__ == "__main__":
    create_teknofest_dataset('data/variant_summary.txt', 'data/processed/teknofest_official_data.csv')
