from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

def get_hybrid_ensemble():
    """
    XGBoost ve LightGBM tabanlı hibrit ensemble mimarisi.
    Teknofest Sağlıkta Yapay Zeka raporu - Bölüm 3.6 - Seçilen Algoritmalar
    ve Gerekçe kısmına uygundur: GBDT algoritma ailesinin en güçlü iki temsilcisinin
    soft-voting ile birleştirilmesi sayesinde varyant profillerindeki genelleme gücü
    ve gürültüye dayanıklılık maksimize edilmiştir.
    """
    
    # XGBoost parametreleri
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # LightGBM parametreleri
    lgbm_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    xgb = XGBClassifier(**xgb_params)
    lgbm = LGBMClassifier(**lgbm_params)
    
    # Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        weights=[0.5, 0.5]
    )
    
    return ensemble
