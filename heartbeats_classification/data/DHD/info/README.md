# Summary ✍🏻
**The dataset contains audio recordings of the heart which have class labels for classification and annotation for heartbeat segmentation. There are several types of heart sounds that can be dangerous symptoms. The dataset contains these types of sounds and labels of their class. The dataset is clean and has no passes. The audio files are of varying lengths, between 1 second and 120 seconds. Total 661 audio files with class label and 390 segmented S1 and S2 heartbeat sound. It can be applied in multi-class classification of various human heart rate abnormalities or heartbeat segmentation.**
****

# Usage 👩🏻‍⚕️
**Having such data, you can set the noble goal of making a  [web application](https://cardio-sonix.streamlit.app/) with a neural network that will analyze the state of your heartbeat over a certain period, which is what I actually did. However, you can improve endlessly, so go for it ✊**
****

# Background 📚
**According to the World Health Organisation, cardiovascular diseases (CVDs) are the number one cause of death globally: more people die annually from CVDs than from any other cause. An estimated 17.1 million people died from CVDs in 2004, representing 29% of all global deaths. Of these deaths, an estimated 7.2 million were due to coronary heart disease. Any method which can help to detect signs of heart disease could therefore have a significant impact on world health. This challenge is to produce methods to do exactly that. Specifically, we are interested in creating the first level of screening of cardiac pathologies both in a Hospital environment by a doctor (using a digital stethoscope) and at home by the patient (using a mobile device).**

**The problem is of particular interest to machine learning researchers as it involves classification of audio sample data, where distinguishing between classes of interest is non-trivial. Data is gathered in real-world situations and frequently contains background noise of every conceivable type. The differences between heart sounds corresponding to different heart symptoms can also be extremely subtle and challenging to separate. Success in classifying this form of data requires extremely robust classifiers. Despite its medical significance, to date this is a relatively unexplored application for machine learning.**
****

# Overview data 🔬
**Data has been gathered from the general public via the iStethoscope Pro iPhone app and from a clinic trial in hospitals using the digital stethoscope DigiScope. The original version of the dataset only had 585 audio samples, but I was able to collect and label 76 audio samples from various sources to increase representativeness and reduce the impact of class imbalance. There are a total of 661 class-labeled audio files in the `audio` folder. The `unlabeled` folder contains 247 audio files that have not been marked up. They are excluded from the markup files to avoid confusing you as they were used to test the model and submit the results. The `info` folder contains only information about the dataset including the current `description`, `license` and `images` with class charts and other information**
## labels
**The `labels.csv` file contains information for each audio recording, namely: the file name, its method of obtaining, as well as the class label. More:**
- **set - get method:** 
    - A - from the general public via the iStethoscope Pro iPhone app 
    - B - from a clinic trial in hospitals using the digital stethoscope DigiScope
    - С - mixed, that is, both A and B categories are possible (obtained and dissected from the Internet)
- **filename - name of the file.**
- **label - class label.**
## annotation
**The `annotation.csv` file contains information for segmenting some audio recordings, namely: filename, cycle, sound, location. More:**
- **sound - type of heartbeat sound (S1 or S2).**
- **filename - name of the file.**
- **cycle - cycle number.**
- **location - the location where the sound is detected.** 
## extra
**Additional data taken from the publicly available [dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease). I express my deep gratitude to the author of the dataset, [Kamil Pytlak
](https://www.kaggle.com/kamilpytlak), for his work. I did label-encoding for convenience. In fact, I added this data only because it is easy to obtain from any user, just like the sound of a heartbeat from any phone. Therefore, they can be used to build an ensemble with a sound model and improve the results. Below is a description of extra data that I also took from the public dataset 👇**
### 2020 annual CDC survey data of 300k+ adults related to their health status
**According to the CDC, heart disease is a leading cause of death for people of most races in the U.S. (African Americans, American Indians and Alaska Natives, and whites). About half of all Americans (47%) have at least 1 of 3 major risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), not getting enough physical activity, or drinking too much alcohol. Identifying and preventing the factors that have the greatest impact on heart disease is very important in healthcare. In turn, developments in computing allow the application of machine learning methods to detect "patterns" in the data that can predict a patient's condition.**
##### I made a more detailed description of each feature in a separate file `README.md` in the `extra` folder.
****

# Dataset structure 🏗
  ```
  ├── DHD
  │   │     
  │   ├── labels.csv
  │   ├── annotation.csv
  │   │     
  │   ├── audio
  │   │   └── file.wav
  │   │     
  │   ├── unlabeled
  │   │   └── audio
  │   │        └── file.wav
  │   │   └── unlabels.csv
  │   │     
  │   ├── extra
  │   │   └── CDC_survey_2020.csv
  │   │   └── README.md
  │   │  
  │   ├── info
  │   │   └── README.md
  │   │   └── LICENSE.txt
  │   │   └── images
  │   │         └── chart.png
  │   │      
  ```
****

# Medicine overview 🫀
![](https://www.youtube.com/watch?v=FtXNnmifbhE)
****

# Overview classes 🧪
**The dataset contains 5 classes of heartbeat sounds. There is an imbalance of classes in the dataset. For this reason, on the left you can see the balanced weights of the classes in the graph on the right. You can see below 👇**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F13000811%2Fffa75b3a1a1d0f74b17caec760b71078%2FScreenshot%20from%202023-07-25%2015-24-09.png?generation=1690287881261729&alt=media)

## Normal
**In the Normal category there are normal, healthy heart sounds. These may contain noise in the final second of the recording as the device is removed from the body. They may contain a variety of background noises (from traffic to radios). They may also contain occasional random noise corresponding to breathing, or brushing the microphone against clothing or skin. A normal heart sound has a clear “lub dub, lub dub” pattern, with the time from “lub” to “dub” shorter than the time from “dub” to the next “lub” (when the heart rate is less than 140 beats per minute). Note the temporal description of “lub” and “dub” locations over time in the following illustration:**

**…lub……….dub……………. lub……….dub……………. lub……….dub……………. lub……….dub…**

**In medicine we call the lub sound "S1" and the dub sound "S2". Most normal heart rates at rest will be between about 60 and 100 beats (‘lub dub’s) per minute. However, note that since the data may have been collected from children or adults in calm or excited states, the heart rates in the data may vary from 40 to 140 beats or higher per minute. Dataset B also contains noisy_normal data - normal data which includes a substantial amount of background noise or distortion. You may choose to use this or ignore it, however the test set will include some equally noisy examples.**
## Murmur
**Heart murmurs sound as though there is a “whooshing, roaring, rumbling, or turbulent fluid” noise in one of two temporal locations: (1) between “lub” and “dub”, or (2) between “dub” and “lub”. They can be a symptom of many heart disorders, some serious. There will still be a “lub” and a “dub”. One of the things that confuses non-medically trained people is that murmurs happen between lub and dub or between dub and lub; not on lub and not on dub. Below, you can find an asterisk `!` at the locations a murmur may be.**

**…lub..!!!!...dub……………. lub..!!!!..dub ……………. lub..!!!!..dub ……………. lub..!!!!..dub …**

**or**

**…lub……….dub…!!!!!!….lub………. dub…!!!!!!….lub ………. dub…!!!!!!….lub ……….dub…!!**

## Extra Heart Sound
**Extra heart sounds can be identified because there is an additional sound, e.g. a “lub-lub dub” or a “lub dub-dub”. An extra heart sound may not be a sign of disease.  However, in some situations it is an important sign of disease, which if detected early could help a person.  The extra heart sound is important to be able to detect as it cannot be detected by ultrasound very well. Below, note the temporal description of the extra heart sounds:**

**...lub.lub..........dub..................... lub. lub..........dub................lub.lub...............dub.......**

**or**

**...lub.......... dub.dub......................lub...........dub.dub......................lub...............dub. dub......**

## Artifact
**In the Artifact category there are a wide range of different sounds, including feedback squeals and echoes, speech, music and noise. There are usually no discernable heart sounds, and thus little or no temporal periodicity at frequencies below 195 Hz. This category is the most different from the others. It is important to be able to distinguish this category from the other three categories, so that someone gathering the data can be instructed to try again.**
## Extrasystole
**Extrasystole sounds may appear occasionally and can be identified because there is a heart sound that is out of rhythm involving extra or skipped heartbeats, e.g. a “lub-lub dub” or a “lub dub-dub”. (This is not the same as an extra heart sound as the event is not regularly occuring.) An extrasystole may not be a sign of disease. It can happen normally in an adult and can be very common in children. However, in some situations extrasystoles can be caused by heart diseases. If these diseases are detected earlier, then treatment is likely to be more effective. Below, note the temporal description of the extra heart sounds:**

**…........lub……….dub………..………. lub. ………..……….dub…………….lub.lub……..…….dub…….**

**or**

**…lub………. dub......………………….lub.…………………dub.dub………………….lub……..…….dub.……**