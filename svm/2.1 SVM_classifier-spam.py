import os
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score

import pickle as pk
import gzip

def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = pk.load(stream)
    stream.close()
    return model


def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    pk.dump(model, stream)
    stream.close()


def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()

    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    return dictionary


def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))   #file_no=702, word_no=3000
    train_labels = np.zeros(len(files))             #file_no=702
    count = 0;
    docID = 0;
    for fil in files:                        #從文件夾中列舉文件檔案
      with open(fil) as fi:                  #開啟目前要處理的文件檔案(命名為fi)
        for i,line in enumerate(fi):         #列舉處理文件檔案中的列編號與列資料
          if i == 2:                         #第三列開始處理
            words = line.split()             #將第三列文字分割
            for word in words:               #針對分割後的文字，逐字檢查
              wordID = 0
              for i,d in enumerate(dictionary): #與整個樣本集的特徵關鍵字比較(3000個)
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)  #累計處理文件中的特徵關鍵字數量
      train_labels[docID] =0;             #處理文件的label建立(0:msg，1:spmsg)
      filepathTokens = fil.split("'\'")      #分割處理文件的檔名
      head_tail = os.path.split(fil)
      lastToken=head_tail[1]
      if lastToken.startswith("spmsg"):          #檢查檔名是否有spmsg(代表為spam)
         train_labels[docID] = 1;               #標籤訂為"1"
         count = count + 1
      docID = docID + 1
    return features_matrix, train_labels

TRAIN_DIR = "train-mails"
TEST_DIR = "test-mails"

dictionary = make_Dictionary(TRAIN_DIR)

print("reading and processing emails from file.")

features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

model = svm.SVC(kernel="rbf", C=100, gamma=0.001)        #C-Support Vector Classification

print("Training model.")
#train model
model.fit(features_matrix, labels)

predicted_labels = model.predict(test_feature_matrix)

print("FINISHED classifying. \n")
print("accuracy score : ",accuracy_score(test_labels, predicted_labels))
