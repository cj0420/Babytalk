from sklearn.ensemble import RandomForestClassifier
import random
import librosa
import numpy as np
import pandas as pd
import os

mfccs = pd.DataFrame()
stfts = pd.DataFrame()

for dir in ['hugging', 'hungry', 'sleepy']:
    folder = os.path.join('./audio/', dir)
    files = os.listdir(folder)

    for file in files[:]:
        if file == '.DS_Store':
            continue
        src = os.path.join(folder, file)
        y, sr = librosa.load(src)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)  # MFCC 변환
        mfcc = mfcc.reshape(1, -1)
        mfcc = pd.DataFrame(mfcc)
        mfccs = pd.concat([mfccs, mfcc])

mfccs = mfccs.fillna(0)

y = np.load('./csv/label.npy')
y = np.argmax(y, axis=1)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(mfccs, y)
feature_importances_ = clf.feature_importances_
feature_importances_
index = np.arange(len(feature_importances_))
df = pd.DataFrame({
    'index': index,
    'value': feature_importances_
})

df.shape
d3 = df.sort_values('value').head(300)
for d in d3.index:
    print(d, end=',')
print()

mfccs = mfccs.reset_index()
mfccs = mfccs.drop(['index'], axis=1)
mfccs.head()
mfccs = mfccs.iloc[:, d3.index]

for dir in ['hugging', 'hungry', 'sleepy']:
    folder = os.path.join('./audio/', dir)
    files = os.listdir(folder)

    for file in files[:]:
        if file == '.DS_Store':
            continue
        src = os.path.join(folder, file)
        y, sr = librosa.load(src)
        stft = np.abs(librosa.stft(y))  # STFT 변환
        stft = stft.reshape(1, -1)
        stft = pd.DataFrame(stft)
        stfts = pd.concat([stfts, stft])

stfts = stfts.fillna(0)

y = np.load('./csv/label.npy')
y = np.argmax(y, axis=1)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(stfts, y)
feature_importances_ = clf.feature_importances_

index = np.arange(len(feature_importances_))

df = pd.DataFrame({
    'index': index,
    'value': feature_importances_
})
d3 = df.sort_values('value', ascending=False).head(500)
for d in d3.index:
    print(d, end=',')
print()

stfts = stfts.reset_index()
stfts = stfts.drop(['index'], axis=1)
stfts.head()

stfts = stfts.iloc[:, d3.index]

audio = pd.concat([stfts, mfccs], axis=1)

df = pd.DataFrame(audio)
df['label'] = y
df.to_csv('./csv/audio.csv', index=False)
