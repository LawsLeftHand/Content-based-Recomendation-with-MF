#!/usr/bin/env python
# coding: utf-8

# In[70]:


from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise.model_selection import GridSearchCV


# In[71]:


import pandas as pd
df_data = pd.read_csv('ratings.dat',sep='::',header = None,usecols=[0,1,2])
df_data
#data = Dataset.load_from_df(df_data, reader)


# In[72]:


reader = Reader(rating_scale=(0, 5), line_format='user item rating')


# In[73]:


data = Dataset.load_from_df(df_data, reader)


# In[90]:



trainset, testset = train_test_split(data, test_size=0.1)


# In[105]:


model = SVD(n_factors=200,n_epochs=20, lr_all=0.01,reg_all=0.1)
model.fit(trainset)


# In[78]:


from collections import defaultdict



def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls




# In[79]:


predictions=model.test(testset)


# In[80]:


from surprise import accuracy
accuracy.rmse(predictions)


# In[81]:


accuracy.mae(predictions)


# In[82]:


precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3)

    # Precision and recall can then be averaged over all users
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))


# In[25]:


import pandas as pd
cb_data = pd.read_csv('D:/genome-scores.csv')
cb_data


# In[26]:


dict1={}
arr=[]
key=cb_data['movieId'][0]
i=0
while i < 12000000:
    while(cb_data['movieId'][i]==key):
        arr.append(round(cb_data['relevance'][i],5))
        i=i+1
    dict1[key]=arr
    #print(dict1[key])
    if(len(dict1[key])!=1128):
        print(len(arr))
        break;
    key=cb_data['movieId'][i]
    #print(key)
    arr=[]
    
        
    


# In[27]:


import numpy as np


# In[28]:


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))



    return 0.5 * cos + 0.5 if norm else cos


# In[94]:


dict2=trainset.ur


# In[84]:


len(dict2)


# In[112]:


def coprediction1(uid,iid):
    iuid=trainset.to_inner_uid(uid)
    #iiid=trainset.to_inner_iid(iid)
    arr=dict1[int(iid)][:170]
    arrm=model.qi[trainset.to_inner_iid(iid)]
    list_new = []
    i=1
    for item in arr:
        list_new.append(item)
    for item in arrm:
        list_new.append(5*round(item,5))
    arr=list_new
    #print(len(arr))
    max=0
    save=[]
    numer=0
    deco=0
    for rec in dict2[iuid]:
        try:
            arr2=dict1[int(trainset.to_raw_iid(rec[0]))][:170]#adjust yourself
        except:
            continue
        arr2m=model.qi[rec[0]]
        list_new = []
        for item in arr2:
            list_new.append(item)
        for item in arr2m:
            list_new.append(round(item,5))
        arr2=list_new
        #print(len(arr2))
        co=cosine_similarity(arr, arr2)
        new=[trainset.to_raw_iid(rec[0]),rec[1],co]
        count=len(save)
        if(len(save)==0):
            save.append(new)
        else:
            loc=count-1
            while(loc>=0 and co>save[loc][2]):
                loc=loc-1
            save.insert(loc+1,new)
            if(len(save))>5: save.pop()
    #print(save)
    for bu in save:  
        numer=numer+bu[2]*bu[1]
        deco=deco+bu[2]
    predict=numer/deco
    return predict


# In[109]:


def prediction(uid,iid):
    iuid=trainset.to_inner_uid(uid)
    #iiid=trainset.to_inner_iid(iid)
    arr=dict1[int(iid)][:170]
    max=0
    save=[]
    numer=0
    deco=0
    for rec in dict2[iuid]:
        arr2=dict1[int(trainset.to_raw_iid(rec[0]))][:170]
        co=cosine_similarity(arr, arr2)
        new=[trainset.to_raw_iid(rec[0]),rec[1],co]
        count=len(save)
        if(len(save)==0):
            save.append(new)
        else:
            loc=count-1
            while(loc>=0 and co>save[loc][2]):
                loc=loc-1
            save.insert(loc+1,new)
            if(len(save))>5: save.pop()
    print(save)
    for bu in save:  
        numer=numer+bu[2]*bu[1]
        deco=deco+bu[2]
    predict=numer/deco
    return predict
        
   


# In[93]:


te_data = pd.read_csv('testing.csv',sep='::',header = None,usecols=[0,1,2])
te_data


# In[32]:


import math


# In[114]:


listre=[]
sum1=0
sum2=0
i=0
for line in testset:
    uid=str(line[0])
    iid=str(line[1])
    rui=line[2]
    try:
        pred = coprediction1(uid,iid)
    except:
        print(line)
    else:
        arrpr=[uid,iid,rui,pred,'im']
        listre.append(arrpr)
        i=i+1
        sum1=sum1+(pred-rui)*(pred-rui)
        sum2=sum2+abs(pred-rui)
    if(i!=0 and i%1000==0):
        RMSE=math.sqrt(sum1/i)
        print(RMSE)
        MAE=sum2/i
        print(MAE)
        precisions, recalls = precision_recall_at_k(listre, k=10, threshold=3)
        print(sum(prec for prec in precisions.values()) / len(precisions))
        print(sum(rec for rec in recalls.values()) / len(recalls))
    if(i>10000):
        break
print(len(listre))        


# In[53]:


precisions, recalls = precision_recall_at_k(listre, k=10, threshold=3)
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))


# In[ ]:




