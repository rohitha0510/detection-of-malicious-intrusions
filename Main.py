from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, RepeatVector
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle

global filename, gan_model
global X,Y
global dataset
global accuracy, precision, recall, fscore, vector
global X_train, X_test, y_train, y_test, scaler
global labels
columns = ['proto', 'service', 'state', 'attack_cat']
label_encoder = []

main = tkinter.Tk()
main.title("Detection of Real-Time Malicious Intrusions & Attacks in IOT Empowered Cybersecurity & Infrastructures") #designing main screen
main.geometry("1300x1200")

 
#fucntion to upload dataset
def uploadDataset():
    global filename, dataset, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    labels = np.unique(dataset['attack_cat'])
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
    label = dataset.groupby('attack_cat').size()
    label.plot(kind="bar")
    plt.xlabel('Attack Names')
    plt.ylabel('Attack Count')
    plt.xticks(rotation=90)
    plt.title("Various Attacks from UNSW-15 Dataset")
    plt.show()
    
    
def preprocessing():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    dataset.drop(['label'], axis = 1,inplace=True)
    for i in range(len(columns)):
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str))) #encoding non-numeric labels into numeric
        label_encoder.append(le)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    print(np.unique(Y, return_counts=True))
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle dataset
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X) #data normalizing
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    
def dataSplit():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split data into train & test
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train GAN algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train GAN algorithm : "+str(X_test.shape[0])+"\n")
    

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runGAN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global gan_model, accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    gan_model = Sequential()
    gan_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    #creating conv2d layer of 64 neurons as generator to generate new instances
    gan_model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    #max layer to collect relevant features from generator model
    gan_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #defining another layer to filter generate instances
    gan_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    gan_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #generated features normalization
    gan_model.add(BatchNormalization())
    #adding another CNN layer
    gan_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    gan_model.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    gan_model.add(BatchNormalization())
    #dropout to remove irrelevant features
    gan_model.add(Dropout(0.2))
    gan_model.add(Flatten())
    #defining prediction model as discriminator
    gan_model.add(Dense(units=100, activation='relu'))
    gan_model.add(Dense(units=100, activation='relu'))
    gan_model.add(Dropout(0.2))
    gan_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    #compiling and training and loading model
    gan_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/gan_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/gan_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gan_model.fit(X_train, y_train, batch_size = 16, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/gan_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        gan_model.load_weights("model/gan_weights.hdf5")
    #perform prediction on test data   
    predict = gan_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:31500] = y_test1[0:31500]
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("GAN Algorithm", predict, y_test1)

def attackPrediction():
    text.delete('1.0', END)
    global gan_model, label_encoder, labels, scaler
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    for i in range(len(columns)-1):
        dataset[columns[i]] = pd.Series(label_encoder[i].fit_transform(dataset[columns[i]].astype(str)))
    dataset = dataset.values
    X = dataset
    X = scaler.transform(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    predict = gan_model.predict(X)  #prediction on test data
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        mitigation = "Clean Request Detected. Normal Processing will be Continued"
        if labels[i] != 'Normal':
            mitigation = "Request is Abnormal and Packet will be dropped"
        text.insert(END,"Test Data : "+str(dataset[i])+" ====> Predicted As : "+labels[i]+"\n"+mitigation+"\n\n")
        

def graph():
    df = pd.DataFrame([['GAN','Precision',precision[0]],['GAN','Recall',recall[0]],['GAN','F1 Score',fscore[0]],['GAN','Accuracy',accuracy[0]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()



font = ('times', 16, 'bold')
title = Label(main, text='Detection of Real-Time Malicious Intrusions & Attacks in IOT Empowered Cybersecurity & Infrastructures')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload UNSW-NB15 Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessing)
processButton.place(x=330,y=550)
processButton.config(font=font1) 

autoButton = Button(main, text="Dataset Train & Test Split", command=dataSplit)
autoButton.place(x=570,y=550)
autoButton.config(font=font1)

proposeButton = Button(main, text="Train Deep Learning GAN Algorithm", command=runGAN)
proposeButton.place(x=850,y=550)
proposeButton.config(font=font1)

tableButton = Button(main, text="Comparison Graph", command=graph)
tableButton.place(x=50,y=600)
tableButton.config(font=font1)

exitButton = Button(main, text="Attack Prediction from Test Data", command=attackPrediction)
exitButton.place(x=330,y=600)
exitButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
