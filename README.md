# Dreem Deep Learning Challenge 2023

This repository contains code for our submission to the *Dreem Machine Learning Challenge*. This challenge was also our final exam for the Machine Learning module at CentraleSupélec. It was organized jointly with the start-up Dreem which develops an EEG sleep headband to detect sleep-related conditions (sleep apnea for example). The goal of the challenge was to correctly predict the sleep phases of 3 test subjects through their night. To create and train our models, Dreem provided us with 7 labelled sleep records.

This challenge was hosted on Kaggle. Our model provided the *3rd best performance*, among more than 40 teams.

## Data Description
**Records**

All of the 10 records contain EEG data (separated in 5 channels, a channel being the signal recorded from an electrode placed at a given position on the headband). In addition, the records also contain data from the 3 axes of the headband's accelerometer. Thus, each record has the following structure:

- Column 0: index of the epoch in the record, the i-th epoch corresponds to the data from t = i * 30 seconds to t = (i+1)*30 seconds.
- Column 1:7500: 1<sup>st</sup> EEG (Frontal-occipital) channel sampled at 250Hz
- Column 7501:15000: 2<sup>nd</sup> EEG (Frontal-occipital) channel sampled at 250Hz
- Column 15001:22500: 3<sup>rd</sup> EEG (Frontal-frontal) channel sampled at 250Hz
- Column 22501:30000: 4<sup>th</sup> EEG (Frontal-occipital) channel sampled at 250Hz
- Column 30001:37500: 5<sup>th</sup> EEG (Frontal-occipital) channel sampled at 250Hz
- Column 37501:39000: X-axis accelerometer channel sampled at 50Hz
- Column 39001:40500: Y-axis accelerometer channel sampled at 50Hz
- Column 40501:42000: Z-axis accelerometer channel sampled at 50Hz  

\
Each line of the record represents 30s of data, which we call an epoch (not to be confused with the epochs when training a ML model !). Each epoch has an identifier represented in the first column. The other columns contain all the datapoints recorded during these 30s. With a sampling rate of 250Hz for the EEG channels and 50Hz for the accelerometer, it follows that data recorded by EEG channels during the epoch is stored in 7500 columns, and the accelormeter data for one axis is stored in 1500 columns. In total, the records have 42001 columns.

\
**Hypnograms (labels)**

The train records labels are stored in the form of an hypnogram. Although it has not been done here, the user could easily plot the labels through time for a given record to see the evolution of the sleep stages through the night (and plot what is called a hypnogram).
The label file contains four columns.
- Column 0: identifier of the record (from 0 to 6)
- Column 1: index of the epoch within the record
- Column 2: global identifier of the epoch
- Column 3: Label/sleep stage (from 0 to 4).

The labels have the following significance:
- Label 0 = Wake: The person is awake.
- Label 1 = N1: The person is drowsy.
- Label 2 = N2: the person is in light sleep.
- Label 3 = N3: the person is in deep sleep.
- Label 4 = REM: the person is in paradoxal sleep, they may dream and/or move.

## Our solution
Our approach was inspired by the work of [Malhotra et al](https://arxiv.org/pdf/2207.07753v3.pdf) <sup>1</sup>.

**Filtering**

EEG data is often quite noisy due to ambiant noise (power line noise at 50Hz for instance) and artifacts (heartbeat, breathing, eye movements); preprocessing is necessary. To do so, we filtered the EEG signals from 0.4Hz to 30Hz to remove low frequency noise and power-line noise. The accelerometer data was filtered between 0.06-0.3Hz and 0.6-1.7Hz to extract breathing and heartbeat.

**Feature extraction**

* First, we compute features on the filtered temporal signals. Each record is divided into windows of 30s/60s/90s which overlap if the windows are long enough (stride = 30s). Statistics like skewness are then computed on each of these windows. Thus, each epochs is used at least 3 times to compute the temporal features; it is a way to deal with class imbalance and perform data augmentation.

* Next, we computed some features based on the frequency spectrum of the epochs. The frequency spectrum can be split into different frequency bands that contain information about the mental state of the patient. For instance, a high alpha band (range 8-12Hz) power is correlated with someone being drowsy or a bit sleepy. Knowing that, the spectral power of each of the frequency band is computed. The ratios of these quantities are used as new features.

* Finally, for each epoch n, we added the features from the epochs n-1 and n+1. Indeed, by using the information contained in the adjacent epochs we take advantage of the continuity of the hypnogram.

**Model**

We chose to use a catboost model since tree-based models generally perform well with EEG data. Other models like random forest or XGBoost were also tested out, but their performances were similar or lower. 

The hyperparameter search was conducted using Optuna.


## How to run our code ?

### Set-up
In the command line, move to this repository using the `cd` command. Depending on whether you want to work with a Virtualenv or a conda environment, follow these steps (replace `<env_name>` by the name of your environment):

*With Virtualenv:*

- Create an empty environment with: `python -m venv <env_name>`

- Activate the environment: `source <env_name>/bin/activate` on Mac, or `<env_name>/Scripts/activate` on Windows.

- Install the packages: `pip install -r requirements.txt`

*With Conda:*

- Create an empty environment with: `conda create --name <env_name> --file requirements.txt`

- Activate the environment: `conda activate <env_name>`

### Code

To run the code, simply use the jupyter notebook main.ipynb. The full execution of the notebook may take up to 3 hours.

## Bibliography

<sup>1</sup> : Malhotra, A., Younes, M., Kuna, S.T., Benca, R., Kushida, C.A., Walsh, J., Hanlon, A., Staley, B., Pack, A.I., Pien, G.W.: Performance of an automated polysomnography scoring system versus computer assisted manual scoring. Sleep 36(4), 573{582} (2013)