import os
from sklearn.model_selection import train_test_split
import feature_extraction as ft
import data_augmentation as da
import numpy as np

#-----------------4-fold-cross validation with data augmentation----------------------#
def make_folds_DA(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode):
    queen_1 = []  
    features_1 = []
    for filename in os.listdir(fold1_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold1_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_1.append(out)
            q =ft.queen_info(filepath)
            queen_1.append(q)
                 
    features_1 = np.asarray(features_1)    
    queen_1 = np.asarray(queen_1)    
    
    queen_2 = []  
    features_2 = []
    for filename in os.listdir(fold2_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold2_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_2.append(out)
            q = ft.queen_info(filepath)
            queen_2.append(q)
            
    features_2 = np.asarray(features_2)    
    queen_2 = np.asarray(queen_2)    
    
    queen_3 = []  
    features_3 = []
    for filename in os.listdir(fold3_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold3_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_3.append(out)
            q = ft.queen_info(filepath)
            queen_3.append(q)
    features_3 = np.asarray(features_3)    
    queen_3 = np.asarray(queen_3)    
    
    queen_4 = []  
    features_4 = []
    for filename in os.listdir(fold4_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold4_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_4.append(out)
            q = ft.queen_info(filepath)
            queen_4.append(q)
    
    features_4 = np.asarray(features_4)    
    queen_4 = np.asarray(queen_4)      
    
    #data augmentation
    features_aug, queen_aug = da.data_augmentation(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks)
    
    X1_train = np.concatenate((features_2, features_3, features_4, features_aug))
    X2_train = np.concatenate((features_1, features_3, features_4, features_aug))
    X3_train = np.concatenate((features_1, features_2, features_4, features_aug))
    X4_train = np.concatenate((features_2, features_3, features_1, features_aug))
    
    Y1_train = np.concatenate((queen_2, queen_3, queen_4, queen_aug))
    Y2_train = np.concatenate((queen_1, queen_3, queen_4, queen_aug))
    Y3_train = np.concatenate((queen_1, queen_2, queen_4, queen_aug))
    Y4_train = np.concatenate((queen_2, queen_3, queen_1, queen_aug))
#    
    
    X1_train = np.concatenate((features_2, features_3, features_4))
    X2_train = np.concatenate((features_1, features_3, features_4))
    X3_train = np.concatenate((features_1, features_2, features_4))
    X4_train = np.concatenate((features_2, features_3, features_1))
    
    Y1_train = np.concatenate((queen_2, queen_3, queen_4))
    Y2_train = np.concatenate((queen_1, queen_3, queen_4))
    Y3_train = np.concatenate((queen_1, queen_2, queen_4))
    Y4_train = np.concatenate((queen_2, queen_3, queen_1))
    
    X1_test = features_1
    X2_test = features_2
    X3_test = features_3
    X4_test = features_4
    
    Y1_test = queen_1
    Y2_test = queen_2
    Y3_test = queen_3
    Y4_test = queen_4
    return X1_train, X2_train, X3_train, X4_train, X1_test, X2_test, X3_test, X4_test, Y1_train, Y2_train, Y3_train, Y4_train, Y1_test, Y2_test, Y3_test, Y4_test

#-----------------4-fold-cross validation without data augmentation----------------------#
def make_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode):
    queen_1 = []  
    features_1 = []
    for filename in os.listdir(fold1_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold1_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_1.append(out)
            q =ft.queen_info(filepath)
            queen_1.append(q)
                 
    features_1 = np.asarray(features_1)    
    queen_1 = np.asarray(queen_1)    
    
    queen_2 = []  
    features_2 = []
    for filename in os.listdir(fold2_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold2_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_2.append(out)
            q = ft.queen_info(filepath)
            queen_2.append(q)
            
    features_2 = np.asarray(features_2)    
    queen_2 = np.asarray(queen_2)    
    
    queen_3 = []  
    features_3 = []
    for filename in os.listdir(fold3_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold3_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_3.append(out)
            q = ft.queen_info(filepath)
            queen_3.append(q)
    features_3 = np.asarray(features_3)    
    queen_3 = np.asarray(queen_3)    
    
    queen_4 = []  
    features_4 = []
    for filename in os.listdir(fold4_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold4_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_4.append(out)
            q = ft.queen_info(filepath)
            queen_4.append(q)
    
    features_4 = np.asarray(features_4)    
    queen_4 = np.asarray(queen_4)      
  
    X1_train = np.concatenate((features_2, features_3, features_4))
    X2_train = np.concatenate((features_1, features_3, features_4))
    X3_train = np.concatenate((features_1, features_2, features_4))
    X4_train = np.concatenate((features_2, features_3, features_1))
    
    Y1_train = np.concatenate((queen_2, queen_3, queen_4))
    Y2_train = np.concatenate((queen_1, queen_3, queen_4))
    Y3_train = np.concatenate((queen_1, queen_2, queen_4))
    Y4_train = np.concatenate((queen_2, queen_3, queen_1))
    
    X1_test = features_1
    X2_test = features_2
    X3_test = features_3
    X4_test = features_4
    
    Y1_test = queen_1
    Y2_test = queen_2
    Y3_test = queen_3
    Y4_test = queen_4
    return X1_train, X2_train, X3_train, X4_train, X1_test, X2_test, X3_test, X4_test, Y1_train, Y2_train, Y3_train, Y4_train, Y1_test, Y2_test, Y3_test, Y4_test 

#-------------70-30 split experiment (3label) --------------------#
          
def make_random_3label(queen_directory, noqueen_directory, pip_directory, n_chunks, mode):
    features_q = []
    queen_q = []  
    for filename in os.listdir(queen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(queen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            q = ft.queen_info(filepath)
            queen_q.append(q)
    
    features_q = np.asarray(features_q)    
    queen_q = np.asarray(queen_q)    
    
    queen_nq = []  
    features_nq = []
    for filename in os.listdir(noqueen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(noqueen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_nq.append(out)
            q = ft.queen_info(filepath)
            queen_nq.append(q)
    
    features_nq = np.asarray(features_nq)    
    queen_nq = np.asarray(queen_nq)   
    
    queen_pip = []  
    features_pip = []
    for filename in os.listdir(pip_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(pip_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_pip.append(out)
            q = ft.queen_info(filepath)
            queen_pip.append(q)
    
    features_pip = np.asarray(features_pip)    
    queen_pip = np.asarray(queen_pip) 
    
    X_q_train, X_q_test, Y_q_train, Y_q_test = train_test_split(features_q, queen_q, test_size=0.3)
    X_nq_train, X_nq_test, Y_nq_train, Y_nq_test = train_test_split(features_nq, queen_nq, test_size=0.3)
    X_p_train, X_p_test, Y_p_train, Y_p_test = train_test_split(features_pip, queen_pip, test_size=0.3)
    
    X_train = np.concatenate((X_q_train, X_nq_train, X_p_train))
    X_test = np.concatenate((X_q_test, X_nq_test, X_p_test))
    Y_train = np.concatenate((Y_q_train, Y_nq_train, Y_p_train))
    Y_test = np.concatenate((Y_q_test, Y_nq_test, Y_p_test))
    return X_train, X_test, Y_train, Y_test

  
#-------------70-30 split experiment (2label) --------------------#
          
def make_random_2label(queen_directory, noqueen_directory,  n_chunks, mode):
    features_q = []
    queen_q = []  
    for filename in os.listdir(queen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(queen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            q = ft.queen_info(filepath)
            queen_q.append(q)
    
    features_q = np.asarray(features_q)    
    queen_q = np.asarray(queen_q)    
    
    queen_nq = []  
    features_nq = []
    for filename in os.listdir(noqueen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(noqueen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_nq.append(out)
            q = ft.queen_info(filepath)
            queen_nq.append(q)
    
    features_nq = np.asarray(features_nq)    
    queen_nq = np.asarray(queen_nq)   
    

    
    X_q_train, X_q_test, Y_q_train, Y_q_test = train_test_split(features_q, queen_q, test_size=0.3)
    X_nq_train, X_nq_test, Y_nq_train, Y_nq_test = train_test_split(features_nq, queen_nq, test_size=0.3)

    X_train = np.concatenate((X_q_train, X_nq_train))
    X_test = np.concatenate((X_q_test, X_nq_test))
    Y_train = np.concatenate((Y_q_train, Y_nq_train))
    Y_test = np.concatenate((Y_q_test, Y_nq_test))
    return X_train, X_test, Y_train, Y_test

#-------------japanese data training--------------------#

def make_only_training(queen_directory, noqueen_directory, n_chunks, mode):
    features_q = []
    queen_q = []  
    for filename in os.listdir(queen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(queen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            q = ft.queen_info(filepath)
            print(q)
            queen_q.append(q)
    
    features_q = np.asarray(features_q)    
    queen_q = np.asarray(queen_q)    
    
    queen_nq = []  
    features_nq = []
    for filename in os.listdir(noqueen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(noqueen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_nq.append(out)
            q = ft.queen_info(filepath)
            queen_nq.append(q)
    
    features_nq = np.asarray(features_nq)    
    queen_nq = np.asarray(queen_nq)   
    print(queen_nq)
    X_train = np.concatenate((features_q, features_nq))
    Y_train = np.concatenate((queen_q, queen_nq))
    print(Y_train)
    
    return X_train, Y_train
    #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9)
    
    #return X_train, X_val, Y_train, Y_val



def make_validation_no_labels(directory, n_chunks, mode):
    features_q = []
    minutes = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            minute = ft.second_info(filename)
            minutes.append(minute)
            filepath = os.path.join(directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            features = np.array(features_q)
            
    
    X_test = np.asarray(features)    
    return X_test, minutes

def make_only_training_3label(queen_directory, noqueen_directory, bees_directory, n_chunks, mode):
    features_q = []
    queen_q = []  
    for filename in os.listdir(queen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(queen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            q = ft.queen_info(filepath)
            print(q)
            queen_q.append(q)
    
    features_q = np.asarray(features_q)    
    queen_q = np.asarray(queen_q)    
    
    queen_nq = []  
    features_nq = []
    for filename in os.listdir(noqueen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(noqueen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_nq.append(out)
            q = ft.queen_info(filepath)
            queen_nq.append(q)
    
    features_nq = np.asarray(features_nq)    
    queen_nq = np.asarray(queen_nq)   
    
    queen_bees = []  
    features_bees = []
    for filename in os.listdir(bees_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(bees_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_bees.append(out)
            q = 2
            queen_bees.append(q)
    
    features_bees = np.asarray(features_bees)    
    queen_bees = np.asarray(queen_bees)   
    
    X_train = np.concatenate((features_q, features_nq, features_bees))
    Y_train = np.concatenate((queen_q, queen_nq, queen_bees))
    print(Y_train)
    
    return X_train, Y_train
    #X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.9)
    
    #return X_train, X_val, Y_train, Y_val



    
    
    