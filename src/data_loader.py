import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class VariantDataProcessor:
    def __init__(self, file_path, chunk_size=100000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.drop_cols = ['CHROM', 'POS', 'BLOSUM62', 'DISTANCE', 'SSR']
        
    def _is_transition(self, ref, alt):
        transitions = [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]
        if pd.isna(ref) or pd.isna(alt): return 0
        return 1 if (ref, alt) in transitions else 0

    def _is_transversion(self, ref, alt):
        transversions = [('A', 'C'), ('C', 'A'), ('A', 'T'), ('T', 'A'),
                         ('G', 'C'), ('C', 'G'), ('G', 'T'), ('T', 'G')]
        if pd.isna(ref) or pd.isna(alt): return 0
        return 1 if (ref, alt) in transversions else 0

    def process_chunk(self, df):
        # 1. Drop useless basic columns
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        # Also drop MOTIF_* columns
        motif_cols = [c for c in df.columns if c.startswith('MOTIF_')]
        df.drop(columns=cols_to_drop + motif_cols, inplace=True, errors='ignore')

        # 2. Transition / Transversion from REF / ALT
        if 'REF' in df.columns and 'ALT' in df.columns:
            df['is_transition'] = df.apply(lambda x: self._is_transition(x['REF'], x['ALT']), axis=1)
            df['is_transversion'] = df.apply(lambda x: self._is_transversion(x['REF'], x['ALT']), axis=1)
            df.drop(columns=['REF', 'ALT'], inplace=True, errors='ignore')

        # 3. AF_ESP, AF_EXAC, AF_TGP -> AF_mean, AF_max
        af_cols = [c for c in ['AF_ESP', 'AF_EXAC', 'AF_TGP'] if c in df.columns]
        if af_cols:
            for c in af_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df['AF_mean'] = df[af_cols].mean(axis=1)
            df['AF_max'] = df[af_cols].max(axis=1)

        # 4. Codons
        if 'Codons' in df.columns:
            df['codon_position'] = df['Codons'].str.extract(r'(\d+)').astype(float)
            df['codon_GC_change'] = df['Codons'].apply(lambda x: 1 if isinstance(x, str) and ('G' in x or 'C' in x) else 0)
            
        # 5. Amino_acids (Polar/Nonpolar/Charged)
        if 'Amino_acids' in df.columns:
            polar = ['S', 'T', 'C', 'Y', 'N', 'Q']
            nonpolar = ['G', 'A', 'V', 'L', 'I', 'M', 'W', 'F', 'P']
            charged = ['D', 'E', 'K', 'R', 'H']
            
            def get_aa_group(aa_str):
                if pd.isna(aa_str): return 'unknown'
                aa = str(aa_str)[-1] # Take the mutated amino acid
                if aa in polar: return 'polar'
                if aa in nonpolar: return 'nonpolar'
                if aa in charged: return 'charged'
                return 'unknown'

            df['amino_group'] = df['Amino_acids'].apply(get_aa_group)
            # One-hot encoding for amino groups
            for grp in ['polar', 'nonpolar', 'charged']:
                df[f'aa_{grp}'] = (df['amino_group'] == grp).astype(int)
            df.drop(columns=['amino_group'], inplace=True, errors='ignore')

        # 6. MC (Molecular Consequence)
        if 'MC' in df.columns:
            df['mc_has_multiple'] = df['MC'].apply(lambda x: 1 if isinstance(x, str) and ',' in x else 0)

        # 7. CADD_PHRED (ensure numeric)
        if 'CADD_PHRED' in df.columns:
            df['CADD_PHRED'] = pd.to_numeric(df['CADD_PHRED'], errors='coerce')

        # 8. Handling Missing Values (<80% drop, else impute)
        missing_ratios = df.isnull().mean()
        cols_to_drop_missing = missing_ratios[missing_ratios > 0.80].index
        df.drop(columns=cols_to_drop_missing, inplace=True, errors='ignore')

        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[f'has_{col}'] = df[col].notnull().astype(int)
                    mode_val = df[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 'unknown'
                    df[col] = df[col].fillna(fill_val)
                else:
                    df[f'has_{col}'] = df[col].notnull().astype(int)
                    df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
                    
        return df

    def run_pipeline(self, output_path):
        print(f"[{self.file_path}] isleniyor... Chunk size: {self.chunk_size}")
        
        # Read file. We handle TSV and CSV based on extension or peek
        sep = '\t' if self.file_path.endswith('.txt') or self.file_path.endswith('.tsv') else ','
        
        first_chunk = True
        total_rows = 0
        
        try:
            for chunk in pd.read_csv(self.file_path, sep=sep, chunksize=self.chunk_size, low_memory=False):
                processed_chunk = self.process_chunk(chunk)
                total_rows += len(processed_chunk)
                
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                
                first_chunk = False
                print(f"Processed {total_rows} rows...")
                
                # Sadece ilk 1 chunk isleyip birakiyoruz hizli test amacli.
                # Tam veri icin break kismini kaldirabilirsiniz.
                break 
                
            print(f"Veri muhendisligi tamamlandi! Kayit yeri: {output_path}")
        except FileNotFoundError:
            print(f"Dosya bulunamadi: {self.file_path}")

if __name__ == "__main__":
    input_file = 'data/variant_summary.txt' # Degisebilir
    output_file = 'data/processed/clean_variants_sample.csv'
    
    loader = VariantDataProcessor(file_path=input_file)
    loader.run_pipeline(output_path=output_file)
