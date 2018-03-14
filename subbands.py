#_*_coding:utf-8_*_
import tensorflow as tf  
from PFDFile import *
from a170113profile import getFileName
import numpy 
#from numpy.random import RandomState  

def rec_pic(ori_pic):#use signal to imply whether you have to transpose the matrix or not(undone)
    col = shape(ori_pic)[0]
    rol = shape(ori_pic)[1]    
    #if signal == 1:#need T    
    ori_trans = ori_pic.T#you should transpose the matrix first,and finally transpose it again to return.
    #else:#no need T
        #ori_trans = ori_pic
    result = zeros((col,64))
    remainder = 64%rol
    quotients = int((64-remainder)/rol)
    for i in range(remainder):
        if i < remainder:
            for j in range((quotients+1)*i,(quotients+1)*(i+1)):
                result[j] = ori_trans[i]
        else:
            for j in range(remainder+j*quotients,quotients+remainder+j*quotients):
                result[j] = ori_trans[i]
    return(result.T)

def numpy_array_stick(path,label,row_size):
    #thresold(for subbands) == (?)28;that is,when rol <= 28,
    #we label it directly as RFI(0);when rol > 28,
    #we use poly_2 or poly_3 to fit the prob
    flist_neg = getFileName(path) 
    file_num_neg = len(flist_neg) 
    if row_size == 96:
        file_num_neg = min(file_num_neg,1000)
    train_size = 0
    init_array = zeros((1,4096))
    # 模拟输入是一个二维数组 
    for j in range(file_num_neg):
        file_situ = path+"/"+flist_neg[j]
        cand = PFD(file_situ)
        subbands = (cand.get_subbands())
        col = shape(subbands)[0]
        rol = shape(subbands)[1]
        pic_sub = np.zeros((64,64))
        sub_temp = np.zeros((64,rol))
        if row >= 28:
            if row_size == 96:#transfer(96,x) to (64,x) in sub_temp
                for i_4 in range(64):
                    temp_sum = 0
                    for j_1 in range(rol):
                        sta = 0
                        if i_4%2 == 0:
                            sta = int(3*i_4/2)
                        else:
                            sta = int(3*(i_4-1)/2)
                        for k_1 in range(sta,sta+3):
                            temp_sum = temp_sum+subbands[k_1][j_1]
                        sub_temp[i_4][j_1] = temp_sum/4  

        if rol < 28:
            continue;
        elif rol >= 28 and rol < 64:
            if row_size == 96:
                pic_sub = rec_pic(sub_temp)
            #use recover_picture() to recover the picture whose rol is less than 64
            elif row_size == 32:
                ori_fir_T = subbands.T
                ori_fir_rec = rec_pic(ori_fir_T)
                ori_sec_T = ori_fir_rec.T
                pic_sub = rec_pic(ori_sec_T)
        elif rol == 64:
            if row_size == 96:
                pic_sub = sub_temp
            elif row_size == 32:
                ori_fir_T = subbands.T
                ori_fir_rec = rec_pic(ori_fir_T)
                pic_sub = ori_fir_rec.T
        train_size = train_size+1
        temp_X = pic_sub.reshape((1,4096))
        if j == 0:
            init_array = temp_X
        else:
            init_array = numpy.vstack((init_array,temp_X))
    #initialize Y label_dataset
    init_label = [[label,1-label] for p in range(train_size)]   
    return(init_array,init_label)
        
