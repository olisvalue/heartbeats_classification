# Summary ✍🏻
**Additional data taken from the publicly available [dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease). I express my deep gratitude to the author of the dataset, [Kamil Pytlak
](https://www.kaggle.com/kamilpytlak), for his work. I did label-encoding for convenience. In fact, I added this data only because it is easy to obtain from any user, just like the sound of a heartbeat from any phone. Therefore, they can be used to build an ensemble with a sound model and improve the results. Below is a description of extra data that I also took from the public dataset 👇**
### 2020 annual CDC survey data of 300k+ adults related to their health status
**According to the CDC, heart disease is a leading cause of death for people of most races in the U.S. (African Americans, American Indians and Alaska Natives, and whites). About half of all Americans (47%) have at least 1 of 3 major risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicators include diabetes status, obesity (high BMI), not getting enough physical activity, or drinking too much alcohol. Identifying and preventing the factors that have the greatest impact on heart disease is very important in healthcare. In turn, developments in computing allow the application of machine learning methods to detect "patterns" in the data that can predict a patient's condition.**
****
# Overview data 🔬
**In total we have 319 795 observations with 18 features.**
* HeartDisease -  respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI).
  -  `No`: 0
  -  `Yes`: 1
* BMI - Body Mass Index (BMI).
  - `weight (kg) / height (m)2.`
* Smoking - Have you smoked at least 100 cigarettes in your entire life?
  -  `No`: 0
  -  `Yes`: 1
* AlcoholDrinking - Do you have more than 14 drinks of alcohol (male) or more than 7 (female) in a week?
  -  `No`: 0
  -  `Yes`: 1
* Stroke - Did you have a stroke?
  -  `No`: 0
  -  `Yes`: 1
* PhysicalHealth - For how many days during the past 30 days was your physical health not good?
  -  `0-30`
* MentalHealth - For how many days during the past 30 days was your mental health not good?
  -  `0-30`
* DiffWalking - Do you have serious difficulty walking or climbing stairs?
  -  `No`: 0
  -  `Yes`: 1
* Sex - Are you a male or a female?
  -  `Female`: 0
  -  `Male`: 1
* AgeCategory - What is your age category? (years)
  -  `18-24`: 0
  -  `25-29`: 1
  -  `30-34`: 2
  -  `35-39`: 3
  -  `40-44`: 4
  -  `45-49`: 5
  -  `50-54`: 6
  -  `55-59`: 7
  -  `60-64`: 8
  -  `65-69`: 9
  -  `70-74`: 10
  -  `75-79`: 11
  -  `80 or older`: 12
* Race - What race are you?
  - `White`: 0
  - `Black`: 1
  - `Asian`: 2
  - `American Indian/Alaskan Native`: 3
  - `Other`: 4
  - `Hispanic`: 5
* Diabetic - Have you ever had diabetes?
  - `No`: 0
  - `Yes`: 1
  - `Yes (during pregnancy)`: 2
  - `No, borderline diabetes`: 3
* PhysicalActivity - Have you played any sports (running, biking, etc.) in the past month?
  -  `No`: 0
  -  `Yes`: 1
* GenHealth - How can you define your general health?
  - `Excellent`: 0
  - `Very good`: 1
  - `Good`: 2
  - `Fair`: 3
  - `Poor`: 4
* SleepTime - How many hours on average do you sleep?
  -  `No`: 0
  -  `Yes`: 1
* Asthma - Do you have asthma?
  -  `No`: 0
  -  `Yes`: 1
* KidneyDisease - Do you have kidney disease?
  -  `No`: 0
  -  `Yes`: 1
* SkinCancer - Do you have skin cancer?
  -  `No`: 0
  -  `Yes`: 1
