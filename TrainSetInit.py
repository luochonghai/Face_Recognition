#_*_coding:utf-8_*_
import tensorflow as tf  
import numpy as np
from PIL import Image

people = 225
def numpy_array_stick(path,types):
    X_data = np.zeros((1,6400))
    Y_datas = [[0 for k in range(people)] for i in range(people*types)]
    for i in range(people*types):
        Y_datas[i][int(i/10)] = 1
    Y_data = np.array(Y_datas)
    Y_data = Y_data.reshape((types*people,people))
    for i in range(1,people+1):
        for j in range(1,1+types):
            temp_X = np.array(Image.open(path+"SS"+str(i)+"_"+str(j)+".tif","r"))
            temp_X = temp_X.reshape((1,6400))

            if(i == 1 and j == 1):
                X_data = temp_X
            else:
                X_data = np.vstack((X_data,temp_X))
    return (X_data,Y_data)



if __name__ == "__main__":
    numpy_array_stick("D:\\FDU\\Template\\FDUROP\\face_detection_and_recognition\\standard2\\")

    