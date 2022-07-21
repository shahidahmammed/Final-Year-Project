# Miss. Shahidah Kunhih Endiape Mammed
# BSc. (Hons) in Computer Science
# TP055203
# Description: Implement the web interface 
# Starting date: 12/06/2022
# Modified date: 18/07/2022

# import all the libraries
import gradio as gr 
from featureExtraction import extractFeatures
import pickle 
from PIL import Image
import os
import soundfile 
import mimetypes 
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# get the accuracy chart
def getChart(name, accuracy): 
    img = Image.open(open("visuals/accuracyScore/{}accuracyTest.png".format(name), "rb")) # auto matically selects Accuracy Testing Score
    if accuracy == "Accuracy Training Score":        
        img = Image.open(open("visuals/accuracyScore/{}accuracyTrain.png".format(name), "rb")) # select Accuracy Training Score
    print(name)
    return img

# get the model
def getmodel(name):
    model = pickle.load(open("model/GradientBoostingClassifier/{}GradientBoostingClassifier_classifier.model".format(name), 'rb')) # load the model
    return model

# function to accept recorded audio to recognize emotion as output and accuracy scores of dataset
def SER(audio, accuracy, numberEmotions): 
    predict = False # only becomes true if audio is submitted
    try: # try to extract features from audio
        print('Recorded audio:', audio.name)  # backend prints name   
        
        if audio.name.find("10") == -1: # converts the new file or recording to have appropriate sample rate
            target_path = audio.name.split('.')[0]+'TEST.wav'
            os.system(f"ffmpeg -y -i {audio.name} -ac 1 -ar 16000 {target_path} -hide_banner -loglevel error")
            audio.name = target_path

        with soundfile.SoundFile(audio.name) as soundFile: # get the sample and turns the audio into digital signals
            audio.name = soundFile.read(dtype="float32")
            sampleRate = soundFile.samplerate
        
        feature = extractFeatures(audio.name, sampleRate, mfcc=True, melSF=True, zcr=True, rms=True, chroma=True, contrast=True, ton=True).reshape(1, -1) # extract features
        print("Features extraction successful with", feature.shape[1], "features") # backend shows extraction numbers
        predict = True
    except:
        emotion = "Error...Please Upload Audio!" # no audio provided

    if numberEmotions == "6 emotions": # if number of emotions selected was 6 emotions
        model = getmodel("AUGEMO6") # get the 6 emotions model
        img = getChart("AUGEMO6", accuracy) # get the chart for 6 emotions
    
    elif numberEmotions == "5 emotions (excluding disgust)":
        model = getmodel("AUGEMO5")
        img = getChart("AUGEMO5", accuracy)
    
    elif numberEmotions == "4 emotions (excluding disgust, neutral)":
        model = getmodel("AUGEMO4")
        img = getChart("AUGEMO4", accuracy)

    else:
        model = getmodel("AUGEMO3")
        img = getChart("AUGEMO3", accuracy)
        
    if predict:
        emotion = model.predict(feature)[0] # model recognizes emotion
        
    return emotion, img # return emotion and accuracy score chart

# inputs
heading = "Speech Emotion Recognition System" # heading in the webpage
userAudio = gr.inputs.Audio(source="upload", type="file", label="Upload to recognize the emotion!") # upload an audio
imageAccuracy = gr.inputs.Radio(["Accuracy Testing Score", "Accuracy Training Score"], label="Select the chart to see CREMA-D accuracy score of diffierent algorithms:") # choose accuracy chart
numberEmotions = gr.inputs.Dropdown(["6 emotions", "5 emotions (excluding disgust)", "4 emotions (excluding disgust, neutral)", "3 emotions (excluding disgust, neutral, angry)"], label="Select the number of emotions:") # choose number of emotions

# outputs
prediction = gr.outputs.Label(label="Emotion Recognized is") # output of uploaded audio
predictionACC = gr.outputs.Image(label="Chart produced") # output of chart selected

# interfaces
description = "A Speech Emotion Recognition for Culturally Diverse English Speakers to recognize Happy, Sad, Fear, Angry, Neutral, and Disgust emotions."
uploadAudio = gr.Interface(fn=SER, inputs=[userAudio, imageAccuracy, numberEmotions], outputs=[prediction, predictionACC], title=heading, description=description, allow_flagging="never") # upload interface
uploadAudio.launch() # launches interface on web-browser
