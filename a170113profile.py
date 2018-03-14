#profile
from PFDFile import *
from sklearn import svm
import os

def getFileName(path):
    flist = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == '.pfd':
            flist.append(i)
    return(flist)

def profile_division_data():
    path_neg = "/home/luzihao/xiaoluo/dataset/bin/p309n_pfd"
    path_pos = "/home/luzihao/xiaoluo/dataset/bin/p309p_pfd"
    flist_neg = getFileName(path_neg)
    flist_pos = getFileName(path_pos)
    file_num_neg = len(flist_neg)
    file_num_pos = len(flist_pos)
    aver_negative = []
    aver_positive = []
    y_nega = [0 for i in range(file_num_neg)]
    y_posi = [1 for i in range(file_num_pos)]

    test_neg = "/home/luzihao/xiaoluo/dataset/bin/RFI_all_info"
    test_pos = "/home/luzihao/xiaoluo/dataset/bin/Good_check_True"
    tlist_neg = getFileName(test_neg)
    tlist_pos = getFileName(test_pos)
    file_test_neg = len(tlist_neg)
    file_test_pos = len(tlist_pos)
    test_div_neg = []
    test_div_pos = []
    y_test_nega = [0 for i in range(file_num_neg)]
    y_test_posi = [1 for i in range(file_num_pos)]

    for j in range(file_num_neg):
        file_situ = path_neg+"/"+flist_neg[j]
        cand = PFD(file_situ)
        profile_list = cand.getprofile()
        peak_prof = max(profile_list)
        aver_prof = sum(profile_list)/len(profile_list) 
        div_nega = peak_prof/aver_prof
        aver_negative.append([div_nega])

    for k in range(file_num_pos):
        file_situ = path_pos+"/"+flist_pos[k]
        cand = PFD(file_situ)
        profile_list = cand.getprofile()
        peak_prof = max(profile_list)
        aver_prof = sum(profile_list)/len(profile_list) 
        div_posi = peak_prof/aver_prof
        aver_positive.append([div_posi])
    
    for s in range(file_test_neg):
        file_situ = test_neg+"/"+tlist_neg[s]
        cand = PFD(file_situ)
        profile_list = cand.getprofile()
        peak_prof = max(profile_list)
        aver_prof = sum(profile_list)/len(profile_list) 
        div_nega = peak_prof/aver_prof
        test_div_neg.append([div_nega])

    for t in range(file_test_pos):
        file_situ = test_pos+"/"+tlist_pos[s]
        cand = PFD(file_situ)
        profile_list = cand.getprofile()
        peak_prof = max(profile_list)
        aver_prof = sum(profile_list)/len(profile_list) 
        div_nega = peak_prof/aver_prof
        test_div_pos.append([div_nega])

    clf = svm.SVC()
    clf.fit(aver_positive+aver_negative,y_posi+y_nega)
    result = clf.predict(test_div_pos+test_div_neg)
    true_res = y_test_posi+y_test_nega
    counter = 0
    for i in range(len(result)):
        if true_res[i] == result[i]:    
            counter = counter+1
    print(counter/len(result))
    
if __name__ == '__main__':
    profile_division_data()
