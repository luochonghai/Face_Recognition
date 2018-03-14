#_*_coding:utf-8_*_
import tensorflow as tf  
from PFDFile import *
from a170113profile import getFileName
import numpy 
#from numpy.random import RandomState  

def rec_pic(ori_pic):
    rol = shape(ori_pic)[1]
    remainder = 64%rol
    quotients = int((64-remainder)/rol)
    result = zeros((64,64))
    ori_trans = ori_pic.T
    for i in range(remainder):
        if i < remainder:
            for j in range((quotients+1)*i,(quotients+1)*(i+1)):
                result[j] = ori_trans[i]
        else:
            for j in range(remainder+j*quotients,quotients+remainder+j*quotients):
                result[j] = ori_trans[i]
    return(result.T)

def numpy_array_stick(path,label,row_size):
    #thresold == 28;that is,when rol <= 28,
    #we label it directly as RFI(0);when rol > 28,
    #we use poly_2 or poly_3 to fit the prob
    flist_neg = getFileName(path) 
    file_num_neg = len(flist_neg)
    if row_size == 256:
        file_num_neg = min(file_num_neg,1000)
        print("file_num_neg for label(%d with 256) is: %d"%(label,file_num_neg))
    train_size = 0
    init_array = zeros((1,4096))
    # 模拟输入是一个二维数组 
    for j in range(file_num_neg):
        file_situ = path+"/"+flist_neg[j]
        cand = PFD(file_situ)
        subints = (cand.get_subints())
        col = shape(subints)[0]
        rol = shape(subints)[1]
        pic_sub = np.zeros((64,64))
        sub_temp = np.zeros((64,rol))
        if rol >= 28:
            if row_size == 256:#from(256,x)->(64,x)
                for i_4 in range(64):
                    temp_sum = 0
                    for j_1 in range(rol):
                        for k_1 in range(i_4*4,(i_4+1)*4):
                            temp_sum = temp_sum+subints[k_1][j_1]
                        sub_temp[i_4][j_1] = temp_sum/4  

        if rol < 28:
            continue;
        elif rol >= 28 and rol < 64:
            if row_size == 256:
                pic_sub = rec_pic(sub_temp)
            #use recover_picture() to recover the picture whose rol is less than 64
            else:
                pic_sub = rec_pic(subints)
        elif rol == 64:
            if row_size == 32 and col != 64:
                continue;
            elif row_size == 256:
                pic_sub = sub_temp
            elif row_size == 32 and col == 64:
                pic_sub = subints
        train_size = train_size+1
        temp_X = pic_sub.reshape((1,4096))
        if j == 0:
            init_array = temp_X
        else:
            init_array = numpy.vstack((init_array,temp_X))
    #initialize Y label_dataset
    init_label = [[label,1-label] for p in range(shape(init_array)[0])]#here exists a bug:train_size != shape(init_array)[0]   
    return(init_array,init_label)
        

if __name__ == "__main__":  
    #定义每次训练数据batch的大小为8，防止内存溢出  
    batch_size = 8 
    #定义神经网络的参数  
    w1 = tf.Variable(tf.random_normal([4096,64],stddev=1,seed=1))  
    w2 = tf.Variable(tf.random_normal([64,1],stddev=1,seed=1))  
    #定义输入和输出  
    x = tf.placeholder(tf.float32,shape=(None,4096),name="x-input")  
    y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")  
    #定义神经网络的前向传播过程  
    a = tf.matmul(x,w1)  
    y = tf.matmul(a,w2)  
    #定义损失函数和反向传播算法  
    #使用交叉熵作为损失函数  
    #tf.clip_by_value(t, clip_value_min, clip_value_max,name=None)  
    #基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况  
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))  
    # 使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小  
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  
    #通过随机函数生成一个模拟数据集  
    #rdm = RandomState(1)  
    # 定义数据集的大小  
    #dataset_size = 128
    D_nega = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309n_pfd",0)
    D_posi = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309p_pfd",1)
    Y_nega = D_nega[1]
    X_nega = D_nega[0]
    Y_posi = D_posi[1]
    X_posi = D_posi[0]
    #创建会话运行TensorFlow程序  
    with tf.Session() as sess: 
        for z in range(2):
            if z == 0:
                Y = Y_nega
                X = X_nega
            else:
                Y = Y_posi
                X = X_posi
        dataset_size = len(Y)  
        #初始化变量  tf.initialize_all_variables()  
        init = tf.initialize_all_variables()  
        sess.run(init)  
        #设置神经网络的迭代次数  
        steps = 500  
        for i in range(steps):  
            #每次选取batch_size个样本进行训练  
            start = (i * batch_size) % dataset_size  
            end = min(start + batch_size,dataset_size)  
            #通过选取样本训练神经网络并更新参数  
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})  
            #每迭代100次输出一次日志信息  
            if i % 100 == 0 :  
                # 计算所有数据的交叉熵  
                total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})  
                # 输出交叉熵之和  
                print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))  
        #输出参数w1  
        print(w1.eval(session=sess))  
        #输出参数w2  
        print(w2.eval(session=sess))  
