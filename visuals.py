# Miss. Shahidah Kunhih Endiape Mammed
# BSc. (Hons) in Computer Science
# TP055203
# visualization 
# Starting date: 07/07/2022
# Modified date: 18/07/2022

# import all the libraries
import pickle
import matplotlib.pyplot as plt
import glob, os
import pandas as pd
import seaborn as sb
from featureExtraction import cremaD
import librosa.display
import numpy as np

# function to save all accuracy charts
def saveCharts():
    bestClassifiers = pickle.load(open("model/AUGEMO6_models.pickle", "rb"))
    accuracyChart(bestClassifiers, name="AUGEMO6", namechart="For 6 Emotions")
    
    bestClassifiers = pickle.load(open("model/AUGEMO5_models.pickle", "rb"))
    accuracyChart(bestClassifiers, name="AUGEMO5", namechart="For 5 Emotions")
    
    bestClassifiers = pickle.load(open("model/AUGEMO4_models.pickle", "rb"))
    accuracyChart(bestClassifiers, name="AUGEMO4", namechart="For 4 Emotions")
    
    bestClassifiers = pickle.load(open("model/AUGEMO3_models.pickle", "rb"))
    accuracyChart(bestClassifiers, name="AUGEMO3", namechart="For 3 Emotions")

# function to save a single accuracy training and testing chart
def accuracyChart(bestClassifiers, name="", namechart=""):
    classifiers = []
    accTr = []
    accTe = []
    for model, accTrain, accTest in bestClassifiers:
        accTr.append(accTrain)
        accTe.append(accTest)
        classifiers.append(model.__class__.__name__)

    # training accuracy score
    plt.figure(figsize=(14, 8))
    plt.rc('font', size=12)
    plt.title("Accuracy Training Dataset Score {}".format(namechart))
    plt.ylabel("Accuracy Score")
    plt.xlabel("Classifiers")
    plt.bar(classifiers, accTr, color=['lightsteelblue', 'slateblue', 'tan', 'salmon', 'seagreen'])
    if not os.path.isdir("visuals/accuracyScore"): # if there is no folder create
        os.mkdir("visuals/accuracyScore")
    plt.savefig("visuals/accuracyScore/{}accuracyTrain.png".format(name)) # saves the plot

    # testing accuracy score
    plt.figure(figsize=(14, 8))
    plt.rc('font', size=12)
    plt.title("Accuracy Testing Dataset Score {}".format(namechart))
    plt.ylabel("Accuracy Score")
    plt.xlabel("Classifiers")
    plt.bar(classifiers, accTe, color=['lightsteelblue', 'slateblue', 'tan', 'salmon', 'seagreen'])
    if not os.path.isdir("visuals/accuracyScore"): # if there is no folder create
        os.mkdir("visuals/accuracyScore")
    plt.savefig("visuals/accuracyScore/{}accuracyTest.png".format(name)) # saves the plot



# function to extract emotion count from dataset
def emotionInfo():
    fileOfEmotions = [] # list to store the emotions
    fileOfPath = [] # list to store the path of corresponding emotions

    for audio in glob.glob("data/AudioWAV/*.wav"): # read from the dataset folder 
            filePath = os.path.basename(audio)
            fileOfPath.append(filePath)
            emotionsFromAudio = cremaD[filePath.split("_")[2]]
            if emotionsFromAudio == "happy":
                fileOfEmotions.append(emotionsFromAudio) # append the emotion into the emotion list
            elif emotionsFromAudio == "sad":
                fileOfEmotions.append(emotionsFromAudio)
            elif emotionsFromAudio == "angry":
                fileOfEmotions.append(emotionsFromAudio)
            elif emotionsFromAudio == "disgust":
                fileOfEmotions.append(emotionsFromAudio)
            elif emotionsFromAudio == "fear":
                fileOfEmotions.append(emotionsFromAudio)
            elif emotionsFromAudio == "neutral":
                fileOfEmotions.append(emotionsFromAudio)

    emotionDF = pd.DataFrame(fileOfEmotions, columns=["Emotion"]) # create dataframe from emotion list
    pathDF = pd.DataFrame(fileOfPath, columns=["Path of Emotions"]) # create dataframe from path list
    cremaDF = pd.concat([emotionDF, pathDF], axis=1)
    cremaDF.head()
    print(cremaDF)

    numberofEmotions = cremaDF.groupby(["Emotion"]).size() # number of each emotions from dataset
    print(numberofEmotions)

    plt.title("Number of Emotions in CREMA-D", size=15) 
    sb.countplot(cremaDF.Emotion)
    plt.ylabel('Count', size=10)
    plt.xlabel('Emotions', size=10)
    sb.despine(top=True, right=True, left=False, bottom=False)

    if not os.path.isdir("visuals"): # if there is no folder create
        os.mkdir("visuals")
    plt.savefig("visuals/number_of_emotions.png") # saves the plot


    # Results
    # Emotion
    # angry      1271
    # disgust    1271
    # fear       1271
    # happy      1271
    # neutral    1087
    # sad        1271


# function to save the feature charts
def savefig(audioPath, name):
    if not os.path.isdir("visuals/{}".format(name)): # if there is no folder create
        os.mkdir("visuals/{}".format(name))
    filePath = os.path.basename(audioPath)
    plt.savefig("visuals/{}/{}.png".format(name, filePath)) # saves the plot

# function to display MFCC of the emotion given
def mfccinfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10,5))
    plt.title("The MFCC for the given audio with {} emotion".format(emotion))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    savefig(audioPath, name="MFCC")

# function to display spectogram of the emotion given
def melinfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    melSF = librosa.feature.melspectrogram(audio, sr=sr)
    plt.figure(figsize=(10,5))
    plt.title("The Mel-Spectrogram for the given audio with {} emotion".format(emotion))
    librosa.display.specshow(melSF, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    savefig(audioPath, name="Mel")

# function to display chroma of the emotion given
def chromainfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    stft = librosa.stft(audio)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    plt.figure(figsize=(10,5))
    plt.title("The Chroma for the given audio with {} emotion".format(emotion))
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    savefig(audioPath, name="Chroma")

# function to display RMS of the emotion given
def rmsinfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    s, ph = librosa.magphase(librosa.stft(audio))
    rms = librosa.feature.rms(S=s)
    fig, ax = plt.subplots(figsize=(15, 6), nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(s, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')    
    ax[0].set(title="The RMS for the given audio with {} emotion".format(emotion))
    savefig(audioPath, name="RMS")

# function to display ZCR of the emotion given
def zcrinfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    plt.figure(figsize=(10,5))
    plt.title("The ZCR for the given audio with {} emotion".format(emotion))
    plt.plot(zcr[0])
    savefig(audioPath, name="ZCR")

# function to display Contrast of the emotion given
def coninfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    stft = librosa.stft(audio)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    plt.figure(figsize=(10,5))
    plt.title("The Constrast for the given audio with {} emotion".format(emotion))
    librosa.display.specshow(contrast, sr=sr, x_axis='time')
    plt.ylabel('Frequency bands')
    plt.colorbar()
    savefig(audioPath, name="Contrast")

# function to display Tonnetz of the emotion given
def toninfo(audioPath):
    audio, sr = librosa.load(audioPath)
    emotion = cremaD[audioPath.split("_")[2]]
    ton = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    plt.figure(figsize=(10,5))
    plt.title("The Tonnetz for the given audio with {} emotion".format(emotion))
    librosa.display.specshow(ton, sr=sr, x_axis='time', y_axis='tonnetz')
    plt.colorbar()
    savefig(audioPath, name="Tonnetz")

# function to produce chart and save for all features
def saveFeatures():
    audioPath = "data/AudioWAV/1001_DFA_ANG_XX.wav"
    mfccinfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_FEA_XX.wav"
    melinfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_HAP_XX.wav"
    chromainfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_SAD_XX.wav"
    rmsinfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_DIS_XX.wav"
    zcrinfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_NEU_XX.wav"
    coninfo(audioPath)

    audioPath = "data/AudioWAV/1001_DFA_ANG_XX.wav"
    toninfo(audioPath) 

    
# execution of functions above
emotionInfo()
saveCharts()
saveFeatures()