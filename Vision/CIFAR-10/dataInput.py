# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:38:00 2018

@author: minsooyeo119112
"""
from matplotlib import pyplot as plt
import numpy as np

class dataInput:
    
    def __init__(self):
        self.trainData, self.testData = self.dataLoad()
        self.labelName = ['airplane',
                          'automobile',
                          'bird',
                          'cat',
                          'deer',
                          'dog',
                          'frog',
                          'horse',
                          'ship',
                          'truck']
        
    def dataLoad(self):
        
        data = []
        for i in range(5):
            dataFrame = self.unpickle('./data_set/cifar_10/data_batch_' + str(i+1))
            data.append(dataFrame)
        testData = self.unpickle('./data_set/cifar_10/test_batch' )
            
        return data, testData
    
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict    
    
    def dataVisuallization(self, label):
        
        tmpDataIdx = [idx for idx, val in enumerate(self.testData[b'labels']) if val ==label]
        tmpData = self.testData[b'data'][tmpDataIdx[np.random.choice(len(tmpDataIdx))],:].reshape([3,32,32])
        plt.imshow(tmpData.transpose(1,2,0))           
        return tmpData

    def dataVisuallizationSubplot(self):
        for i in range(10):
            for q in range(10):
                plt.subplot(10,10,((i)*10) + (q+1) )
                self.dataVisuallization(q)
                if(i==0):
                    plt.title(self.labelName[q])
        

if __name__ == '__main__':
    dataOb = dataInput()

    
    print('main')