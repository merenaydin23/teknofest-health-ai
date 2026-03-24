import joblib
import pandas as pd
import os

def predict(features):
    """
    Kanser tespit tahmin fonksiyonu (Hibrit Ensemble modeli ile).
    
    Args:
        features (list): 9 adet sayısal özellik listesi [Feature_1...Feature_9]
        
    Returns:
        prob (float): Kanser (Pathogenic) olasılığı
        label (str): Tahmin sonucu (BENIGN / PATHOGENIC)
    """
    # Model yolu
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hybrid_ensemble_model.pkl')
    if not os.path.exists(model_path):
        # Local runner için alternatif
        model_path = 'models/hybrid_ensemble_model.pkl'
        
    model = joblib.load(model_path)
    
    # Feature_1...Feature_9 isimleriyle kolonları uyduruyoruz
    df = pd.DataFrame([features], columns=[f'Feature_{i+1}' for i in range(len(features))])
    
    prob = model.predict_proba(df)[0, 1]
    
    # Raporumuzdaki en iyi threshold değeri 0.74 idi
    label_num = 1 if prob >= 0.74 else 0
    label_str = "PATHOGENIC (Patolojik)" if label_num == 1 else "BENIGN (Zararsız/Normal)"
    
    return prob, label_str

if __name__ == "__main__":
    # Örnek kullanım (Random Test Verisi)
    sample_data = [15042, 1, 3, 2, 7360, 9907, 12, 5, 0]
    
    prob, label = predict(sample_data)
    print(f"\n--- TEKNOFEST KANSER TESPIT TAHMINI ---")
    print(f"Olasilik: %{prob*100:.2f}")
    print(f"Sonuc  : {label}")
    print("-" * 38)
