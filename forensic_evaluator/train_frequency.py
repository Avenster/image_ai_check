# train_frequency.py
import os, numpy as np, cv2, joblib
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_features(img):
    # same as in detector
    ...

real_paths = [...]  # list of real image paths
fake_paths = [...]  # list of fake image paths
X, y = [], []
for p in real_paths:
    X.append(extract_features(Image.open(p).convert('RGB')).flatten())
    y.append(0)
for p in fake_paths:
    X.append(extract_features(Image.open(p).convert('RGB')).flatten())
    y.append(1)
X = np.array(X)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
svm = SVC(probability=True).fit(X_scaled, y)
joblib.dump(svm, 'weights/frequency_svm.pkl')
joblib.dump(scaler, 'weights/frequency_svm_scaler.pkl')
