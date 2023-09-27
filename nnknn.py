import math

import numpy as np
import cv2
from statistics import mode
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import matplotlib.pyplot as plt

path = "./ORL/s"

nr_antrenare = 40
nr_poze = 8
mostra = cv2.imread("./ORL/s1/9.pgm",0)
A = np.array([0])
T = np.array([0])
iterator = 0
timpi = []

A, T = [], []
for i in range(1, nr_antrenare+1):
    file_path = path + str(i) + '/'
    for j in range(1, nr_poze+1):
        poza = cv2.imread(file_path + str(j) + '.pgm', 0)
        poza = poza.reshape(-1, 1)
        A.append(poza)
    for t in range(11-nr_poze,11):
        poza = cv2.imread(file_path + str(t) + '.pgm', 0)
        poza = poza.reshape(-1, 1)
        T.append(poza)
A = np.hstack(A)
T = np.hstack(T)

def NN(training_matrix,test_picture,norma=None):
    z=np.zeros(len(A[0]))
    if norma == 'cos':
        for i in range(400-((10-nr_poze)*40)):
            z[i] = 1 - np.dot(training_matrix[:, i], test_picture.reshape(10304,1)) / (np.linalg.norm(training_matrix[:, i].reshape(10304,1)) * np.linalg.norm(test_picture.reshape(10304,1)))

    else:
        for i in range(400-((10-nr_poze)*40)):
            z[i] = np.linalg.norm(training_matrix[:, i].reshape(10304,1) - test_picture.reshape(10304,1), ord=norma)
    pos = np.argmin(z)
    person = int(pos/8) + 1
    return person

def KNN(training_matrix,test_picture,k, norma=None):
   z=np.zeros(len(A[0]))
   if norma == 'cos':
       for i in range(400-((10-nr_poze)*40)):
           z[i] = 1 - np.dot(training_matrix[:, i], test_picture.reshape(10304,1)) / (np.linalg.norm(training_matrix[:, i].reshape(10304,1)) * np.linalg.norm(test_picture.reshape(10304,1)))
   else:
       for i in range(400-((10-nr_poze)*40)):
           z[i] = np.linalg.norm(training_matrix[:, i].reshape(10304,1) - test_picture.reshape(10304,1), ord=norma)
   z = np.argsort(z)

   indexes = np.zeros(k)
   for i in range(k):
       indexes[i] = int(z[i]/nr_poze + 1)

   return int(mode(indexes))

def test_nn(norma=None):
    recognition_rate, execution_time = 0, 0
    for i in range(80):
        start_time = time.time()
        returned_image = NN(A, T[:, i], norma)
        end_time = time.time()
        execution_time += end_time - start_time

        person = (i // 2) + 1
        recognition_rate += returned_image == person

    recognition_rate = recognition_rate / 80
    execution_time = execution_time / 80
    return recognition_rate, execution_time

def testerKNN(k, norma=None):
    recognitionRate = 0
    executionTime = 0
    for i in range(0, 80):
        st = time.time()
        returnedImage = KNN(A, T[:,i], k, norma)
        et = time.time()
        executionTime += (et - st)

        person = int(i / 2) + 1
        if( returnedImage == person):
            recognitionRate += 1

    recognitionRate /= 80
    executionTime /= 80
    return recognitionRate,executionTime

def plot_graph(csv_file):
    df = pd.read_csv(csv_file)
    rec_rate = df['RecRate']
    exec_time = df['ExecTime']
    modes = ['inf', '1', '2', 'cos']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(csv_file)
    axs[0].bar(modes, rec_rate)
    axs[1].bar(modes, exec_time)

    axs[0].set_xlabel('Norm')
    axs[1].set_xlabel('Norm')
    axs[0].set_ylabel('RecRate')
    axs[1].set_ylabel('ExecTime')

    plt.show()

csvFiles = ['knn3.csv', 'knn5.csv', 'knn7.csv','nn.csv']
for i in csvFiles:
    plot_graph(i)