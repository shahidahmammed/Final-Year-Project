# Miss. Shahidah Kunhih Endiape Mammed
# BSc. (Hons) in Computer Science
# TP055203
# Description: Data augmentation nd feature extraction from the audio files
# Starting date: 06/06/2022
# Modified date: 18/07/2022

# import all the libraries
from msilib.schema import ListBox
import soundfile 
import numpy as np
import librosa # librosa has the features for extraction
import os
import glob 
from sklearn.model_selection import train_test_split
import warnings

cremaD = { # the 6 emotions available in CREMA-D identified in a dictionary
    "ANG" : "angry",
    "DIS" : "disgust",
    "FEA" : "fear",
    "HAP" : "happy",
    "NEU" : "neutral",
    "SAD" : "sad"
}

checkEmotions = { # the emotions to recognize from the dataset
    "happy",
    "sad",
    'fear',
    'angry',
    'neutral',
    'disgust'
}

def extractFeatures(chosenFile, sampleRate, **kwargs): # features will be extracted from the audio file 
    warnings.filterwarnings('ignore') # ignore warning signs for clear outputs

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    melSF = kwargs.get("melSF")
    contrast = kwargs.get("contrast")
    ton = kwargs.get("ton")
    rms = kwargs.get("rms")
    zcr = kwargs.get("zcr")    
        
    result = np.array([])
    
    if chroma or contrast:
        stft = np.abs(librosa.stft(chosenFile))        
    
    if mfcc:
        mfccAll = np.mean(librosa.feature.mfcc(y=chosenFile, sr=sampleRate, n_mfcc=40).T, axis=0) # extract MFCC from audio
        result = np.hstack((result, mfccAll)) # add MFCC feature to result horizontally
    
    if melSF:
        melSF = np.mean(librosa.feature.melspectrogram(chosenFile, sr=sampleRate,).T, axis=0) # extract mel-spectogram from audio
        result = np.hstack((result, melSF)) # add mel-specogram feature to result horizontally
    
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T, axis=0) # extract chroma from audio
        result = np.hstack((result, chroma)) # add chroma feature to result horizontally

    if ton:
        ton = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(chosenFile), sr=sampleRate).T, axis=0) # extract ton from audio
        result = np.hstack((result, ton)) # add tonnetz feature to result horizontally
    
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T, axis=0) # extract contrast from audio
        result = np.hstack((result, contrast)) # add contrast feature to result horizontally

    if rms:
        rms = np.mean(librosa.feature.rms(y=chosenFile).T, axis=0)
        result = np.hstack((result, rms)) # add contrast feature to result horizontally
    
    if zcr:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=chosenFile).T, axis=0)
        result = np.hstack((result, zcr)) # add contrast feature to result horizontally

    return result


def dataSet(testSize=0.2, **kwargs): # split the data 25% for the training
    noise = kwargs.get("noise")

    featuresAdded = [] # stores the features from audio
    emotionsAdded = [] # stores the emotions from audio

    for audio in glob.glob("data/AudioWAV/*.wav"): # read from the dataset folder 
        filePath = os.path.basename(audio) # get the filename
        emotionsFromAudio = cremaD[filePath.split("_")[2]] # extract emotion label from file

        with soundfile.SoundFile(audio) as soundFile:
            chosenFile = soundFile.read(dtype="float32")
            sampleRate = soundFile.samplerate
    
        if emotionsFromAudio not in checkEmotions: # only use emotions stated 
            continue

        if noise:
            featuresFromAudio = getFeatures(chosenFile, sampleRate) # features are extracted            
            for e in featuresFromAudio: # for each augmentated file
                featuresAdded.append(e) # appended to the features to feature list
                emotionsAdded.append(emotionsFromAudio) # appended to the emotion to emotion list
        else:
            featuresFromAudio = extractFeatures(chosenFile, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # 7 features are extracted

            featuresAdded.append(featuresFromAudio) # appended to the features to feature list
            emotionsAdded.append(emotionsFromAudio) # appended to the emotion to emotion list

    
    return train_test_split(np.array(featuresAdded), emotionsAdded, test_size=testSize, random_state=5) # splid the data into training and testing


def getFeatures(chosenFile, sampleRate): # get the features from original and augmented data
          
    featuresFromAudio = extractFeatures(chosenFile, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # features are extracted
    AllfeaturesFromAudio = np.array(featuresFromAudio) # added to an array

    noise = noiseInjection(chosenFile)
    noiseFeatures = extractFeatures(noise, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # noise audio features are extracted
    AllfeaturesFromAudio = np.vstack((AllfeaturesFromAudio, noiseFeatures)) # vertically stacking noise features

    stretch = stretching(chosenFile)
    pitchStretch = pitch(stretch, sampleRate)
    PSFeatures = extractFeatures(pitchStretch, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # stretch and pitched features are extracted
    AllfeaturesFromAudio = np.vstack((AllfeaturesFromAudio, PSFeatures)) # vertically stacking stretch and pitched features

    shift = shifting(chosenFile)
    shiftFeatures = extractFeatures(shift, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # shifting audio features are extracted
    AllfeaturesFromAudio = np.vstack((AllfeaturesFromAudio, shiftFeatures)) # vertically stacking shifting features

    sp = speed(chosenFile)
    spFeatures = extractFeatures(sp, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True) # speeded audio features are extracted
    AllfeaturesFromAudio = np.vstack((AllfeaturesFromAudio, spFeatures)) # vertically stacking speeded features

    return AllfeaturesFromAudio # return the all the features


# function to augment the data through noise injection
def noiseInjection(audio):
    noiseAMP = 0.035*np.random.uniform()*np.amax(audio) # audio of noise
    audio = audio + noiseAMP*np.random.normal(size=audio.shape[0]) # new audio with noise injection on the original audio
    return audio

# function to augment the data through changing the pitch
def pitch(audio, sr, pitchFactor=0.7):
    return librosa.effects.pitch_shift(audio, sr, pitchFactor) # new audio with changed pitch

# function to augment the data through stretching the audio
def stretching(audio, rate=0.8):
    return librosa.effects.time_stretch(audio, rate) # new audio that is stretch

# function to augment the data though shifting the audio
def shifting(audio):
    range = int(np.random.uniform(low=5, high=5)*1000) # new audio that is shifted
    return np.roll(audio, range)

# function to augment the data though speeding the audio
def speed(audio):
    length = np.random.uniform(low=0.8, high=1) # get a random uniform number between 0.8 and 1
    speed = 1.2 / length # create the speed
    temp = np.interp(np.arange(0,len(audio),speed),np.arange(0,len(audio)),audio) 
    minlen = min(audio.shape[0], temp.shape[0])
    audio *= 0
    audio[0:minlen] = temp[0:minlen] 
    return audio