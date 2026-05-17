"""
Movie Genre Multi-Label Classification — Optimized Pipeline
Metric: Macro ROC-AUC (Kaggle competition)
Baseline: CountVectorizer(1000) + OneVsRest(RandomForest) → 0.7812
"""

import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

# ── 0. NLTK assets ─────────────────────────────────────────────────────────────
for resource in ('stopwords', 'wordnet', 'omw-1.4', 'punkt'):
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ── 1. ADVANCED TEXT PREPROCESSING ────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Pipeline:
      1. Lowercase
      2. Strip HTML tags
      3. Remove non-alphabetic characters
      4. Tokenize, remove stop-words, lemmatize
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)                    # remove HTML
    text = re.sub(r'[^a-z\s]', ' ', text)                   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── 2. DATA LOADING ────────────────────────────────────────────────────────────
print("Loading data …")
dataTraining = pd.read_csv(
    'https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip',
    encoding='UTF-8', index_col=0
)
dataTesting = pd.read_csv(
    'https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip',
    encoding='UTF-8', index_col=0
)

print(f"Training samples : {len(dataTraining)}")
print(f"Testing  samples : {len(dataTesting)}")

# ── 3. PREPROCESS PLOTS ────────────────────────────────────────────────────────
print("Preprocessing text …")
dataTraining['plot_clean'] = dataTraining['plot'].apply(clean_text)
dataTesting['plot_clean']  = dataTesting['plot'].apply(clean_text)

# ── 4. TARGET ENCODING ────────────────────────────────────────────────────────
dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))
mlb = MultiLabelBinarizer()
y_genres = mlb.fit_transform(dataTraining['genres'])
print(f"Genre classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

# ── 5. FEATURE ENGINEERING — TF-IDF with bigrams ─────────────────────────────
tfidf = TfidfVectorizer(
    max_features=15_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,        # log(1 + tf) dampening
    min_df=2,                 # ignore very rare terms
    analyzer='word',
    token_pattern=r'\b[a-z][a-z]+\b'
)

X_dtm = tfidf.fit_transform(dataTraining['plot_clean'])
print(f"Feature matrix shape: {X_dtm.shape}")

# ── 6. TRAIN / VALIDATION SPLIT ───────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_dtm, y_genres, test_size=0.20, random_state=42
)

# ── 7. MODEL — OneVsRest + Logistic Regression (L2) ──────────────────────────
#
# Why LR instead of RF for sparse TF-IDF features?
#   • RF builds trees on random feature subsets → inefficient on very sparse,
#     high-dim text matrices (many zero entries).
#   • LR (L2) operates on the raw dot-product, leveraging sparsity efficiently
#     and delivers top-tier performance on NLP benchmarks with much less compute.
#
lr = LogisticRegression(
    C=4.0,               # inverse regularization strength (tuned heuristic)
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf = OneVsRestClassifier(lr, n_jobs=-1)

print("Training Logistic Regression (OneVsRest) …")
clf.fit(X_train, y_train)

# ── 8. VALIDATION ─────────────────────────────────────────────────────────────
y_val_pred = clf.predict_proba(X_val)
val_roc = roc_auc_score(y_val, y_val_pred, average='macro')
print(f"\n{'='*50}")
print(f"  Validation Macro ROC-AUC : {val_roc:.4f}  (baseline = 0.7812)")
print(f"{'='*50}\n")

# ── 9. RETRAIN ON FULL TRAINING SET ───────────────────────────────────────────
print("Retraining on full dataset …")
clf_final = OneVsRestClassifier(
    LogisticRegression(C=4.0, max_iter=1000, solver='lbfgs',
                       class_weight='balanced', random_state=42, n_jobs=-1),
    n_jobs=-1
)
clf_final.fit(X_dtm, y_genres)

# ── 10. PREDICT ON KAGGLE TEST SET ────────────────────────────────────────────
# IMPORTANT: use .transform() (not .fit_transform()) to apply the same vocab
X_test_dtm = tfidf.transform(dataTesting['plot_clean'])
print(f"Test feature matrix shape : {X_test_dtm.shape}  (must match train cols)")
assert X_test_dtm.shape[1] == X_dtm.shape[1], "Feature dimension mismatch!"

y_pred_test = clf_final.predict_proba(X_test_dtm)

# ── 11. BUILD SUBMISSION FILE ──────────────────────────────────────────────────
cols = [
    'p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy',
    'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family', 'p_Fantasy',
    'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical',
    'p_Mystery', 'p_News', 'p_Romance', 'p_Sci-Fi', 'p_Short', 'p_Sport',
    'p_Thriller', 'p_War', 'p_Western'
]

# Sanity check: MLB classes must align with cols
expected_genres = [c.replace('p_', '') for c in cols]
assert list(mlb.classes_) == expected_genres, (
    f"Genre mismatch!\n  Expected: {expected_genres}\n  Got: {list(mlb.classes_)}"
)
assert y_pred_test.shape == (len(dataTesting), len(cols)), (
    f"Prediction shape mismatch: {y_pred_test.shape}"
)

res = pd.DataFrame(y_pred_test, index=dataTesting.index, columns=cols)
output_path = 'pred_genres_text_LR_tfidf.csv'
res.to_csv(output_path, index_label='ID')

print(f"\nSubmission saved → {output_path}")
print(res.head())
