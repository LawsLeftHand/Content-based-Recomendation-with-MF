#!/usr/bin/env python
# coding: utf-8

# In[2]:


from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd


# In[3]:


d_data = pd.read_csv('mytest1.csv',sep=',')


# In[5]:


d_data1 = pd.read_csv('CellPhonesAndAccessoriesReview.csv',sep=',')


# In[6]:


d_data=pd.concat([d_data,d_data1],ignore_index=True)


# In[7]:


df_data1 = pd.read_csv('SportsAndOutdoorsReview.csv',sep=',')
d_data=pd.concat([d_data,df_data1],ignore_index=True)


# In[8]:


df_data1 = pd.read_csv('ToysAndGamesReview.csv',sep=',')
d_data=pd.concat([d_data,df_data1],ignore_index=True)


# In[ ]:





# In[9]:


reader = Reader(rating_scale=(1, 5), line_format='item user rating')
data = Dataset.load_from_df(d_data, reader)
data


# In[10]:


trainset = data.build_full_trainset()


# In[17]:


# Or load my trainset directly

#pkl_filename = "D:/pickle_set.pkl"  
#with open(pkl_filename, 'rb') as file:  
 #   trainset = pickle.load(file)


# In[13]:


model = SVD(n_factors=200, n_epochs=20, lr_all=0.01,reg_all=0.1)
model.fit(trainset)


# In[21]:


testli = pd.read_csv('testset.csv',sep=',')


# In[27]:


testlist=[]
for i in range(len(testli)):
    testlist.append(testli['asin'][i])


# In[30]:


# 
# pholist=[]
#for item in testlist:
 #   try:
#        test=trainset.to_inner_uid(item)
#        pholist.append(item)
#    except:
#        continue
#len(pholist)
# 


# In[45]:


#SVD 

dict2={}

for i in range(1,9):
    arr1=[]
    iid=i
    key=iid
    for item in itemlist:
        pred=model.predict(item,iid).est
        temp=[item,pred]
        if(pred>4):
            if(len(arr1)==0):
                arr1.append(temp)
                continue
            loc=len(arr1)-1
            if(pred > arr1[loc][1]):
                arr1.append(0)
                while loc>=0 and arr1[loc][1] < pred:
                    arr1[loc+1]=arr1[loc]
                    loc-=1
                arr1[loc+1]=temp
            else:
                arr1.append(temp)
           # if(len(arr1)>200):
             #   arr1.pop()   
    #print(arr)
    dict2[key]=arr1


# In[50]:


len(dict2[4])


# In[40]:


def prediction(uid,iid):
    iiid=trainset.to_inner_iid(iid)
    #iiid=trainset.to_inner_iid(iid)
    try:
        arr=dict1[uid]
    except:
        return 0
    max=0
    save=[]
    numer=0
    deco=0
    for rec in dict3[iiid]:
        try:
            arr2=dict1[trainset.to_raw_uid(rec[0])]
        except:
            continue
        co=len([l for l in arr if l in arr2])
        new=[trainset.to_raw_uid(rec[0]),rec[1],co]
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
    if(deco==0): return 0
    predict=numer/deco
    return predict


# In[40]:


pkl_filename = "pickle_feature.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(dict1, file)
#Same as trainset, features are divded into 3 parts to make test easier


# In[51]:


import csv
out = open('recommendationtest.csv', 'w',newline='')
csv_write = csv.writer(out,dialect='excel')
csv_write.writerow(['user_id', 'rclist'])
length=len(trainset.ur)
for key in dict2:
    arr=[]
    for item in dict2[key]:
        pred=prediction(item[0],key)
        temp=[item[0],pred]
        if(len(arr)==0):
            arr.append(temp)
        else:
            loc=len(arr)-1
            if(pred > arr[loc][1]):
                arr.append(0)
                while loc>=0 and arr[loc][1] < pred:
                    arr[loc+1]=arr[loc]
                    loc-=1
                arr[loc+1]=temp
            else:
                arr.append(temp)
            if(len(arr)>20):
                arr.pop()
    #print(arr)
    if(len(arr)!=0):
        result=str(arr[0][0])
    for i in range(len(arr)-1):
        result=result+ '::'+ str(arr[i+1][0])
#    print(result)
    csv_write.writerow([key,result])
                
    
    


# In[ ]:


dict3=trainset.ir


# In[46]:


out = open('simitem.csv', 'w',newline='')
csv_write = csv.writer(out,dialect='excel')
csv_write.writerow(['item_id', 'simlist'])
for item in elclist:
    arr=dict1[item]
    save=[]
    for coitem in elclist:
        if(coitem==item):
            continue
        arr2=dict1[coitem]
        co=len([l for l in arr if l in arr2])
        new=[coitem,co]
        count=len(save)
        if(len(save)==0):
            save.append(new)
        else:
            loc=count-1
            while(loc>=0 and co>save[loc][1]):
                loc=loc-1
            save.insert(loc+1,new)
            if(len(save))>20: save.pop()
    if(len(save)!=0):
        result=str(save[0][0])
    for i in range(len(save)-1):
        result=result+ '::'+ str(save[i+1][0])
    print(result)
    csv_write.writerow([item,result])
        


# In[ ]:




