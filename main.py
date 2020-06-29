import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import sys
from pprint import pprint



def get_total_variance(df):
    number = df['Class']
    variance = 1
    temp1 = (df['Class'])
    for p in [0,1]:
        count1 = 0
        for j in range(len(number)):
            if number.iloc[j] == p:
               count1 +=1
        frac = count1/(len(number)+eps)
    variance *= frac
    return variance

def get_prob_variance(df,val):
    number = len(df['Class'])
    init_variance = 1
    out_variance = 0
    for i in [0,1]:
        init_variance = 1
        for j in [0,1]:
            count = 0
            count2 = 0
            temp = df[val]
            temp1 = df['Class']
            for k in range(number):
                if temp.iloc[k] == i and temp1.iloc[k] == j:
                    count += 1
                if temp.iloc[k] == i:
                    count2 += 1

            num = count
            den = count2
           # print("this is 1 for ",i,count)
            #print("this is 2 for ",j,count2)
            frac = num/(den+eps)
            init_variance *= frac
            temp = init_variance
           # print(init_entropy)
        frac2=den/(len(df)+eps)
        out_variance += -(frac2*temp)
    #print(abs(out_variance))
    return abs(out_variance)




def get_total_entropy(df):
    number = df['Class']
    entropy = 0
    temp1 = (df['Class'])
    for p in [0,1]:
        count1 = 0
        for j in range(len(number)):
            if number.iloc[j] == p:
               count1 +=1
        frac = count1/(len(number)+eps)
    entropy += -frac*log(frac)
    return entropy

def get_entropy_label(df,val):
    #elements = df[label].unique()
    number = len(df['Class'])
    init_entropy = 0
    out_entropy = 0
    for i in [0,1]:
        init_entropy = 0
        for j in [0,1]:
            count = 0
            count2 = 0
            temp = df[val]
            temp1 = df['Class']
            for k in range(number):
                if temp.iloc[k] == i and temp1.iloc[k] == j:
                    count += 1
                if temp.iloc[k] == i:
                    count2 += 1

            num = count
            den = count2
           # print("this is 1 for ",i,count)
            #print("this is 2 for ",j,count2)
            frac = num/(den+eps)
            init_entropy += -(frac*log(frac+eps))
            temp = init_entropy
           # print(init_entropy)
        frac2=den/(len(df)+eps)
        out_entropy += -(frac2*temp)
    return abs(out_entropy)


def get_info_gain(df):
    #df1= pd.read_csv(r"C:\Users\Sagar\Desktop\ML\dataset1\training_set.csv")
    val = df.keys()[:-1]
    info_gain = []
    for u in val:
        info_gain.append(get_total_entropy(df)-get_entropy_label(df,u))
    #print(val[info_gain.index(max(info_gain))])
    final = val[info_gain.index(max(info_gain))]
    return final


def get_variance_impurity(df):
   # df1= pd.read_csv(r"C:\Users\Sagar\Desktop\ML\dataset1\training_set.csv")
    val = df.keys()[0:20]
    i_g = []
    for u in val:
        i_g.append(get_total_variance(df)-get_prob_variance(df,u))
    #print(val[info_gain.index(max(info_gain))]
    final1 = val[i_g.index(max(i_g))]
    return final1




def counter(values):
    get_zero = values[values['Class'] == 0]
    count_zero = get_zero['Class'].count()
    get_one = values[values['Class'] == 1]
    count_one = get_one['Class'].count()
    total_count = []
    if (count_zero > 0) & (count_one > 0):
        total_count = [count_zero,count_one]
    elif (count_zero > 0) & (count_one == 0):
        total_count = [count_zero]
    elif (count_zero == 0) & (count_one > 0):
        total_count = [count_one]
    return total_count




def prediction_for_accuracy(labels,tree,default = 1):
    for i in list(labels.keys()):
        if i in list(tree.keys()):
            prediction = tree[i][labels[i]]
            if isinstance(prediction,dict):
                return prediction_for_accuracy(labels,prediction)
            else:
                return prediction




def accuracy(df,tree):
    retrived = (df).iloc[:,:-1].to_dict(orient = "records")
    accuracy = pd.DataFrame(columns=["get_predicted"])
    for i in range(len(df)):
        accuracy.loc[i,"get_predicted"] = prediction_for_accuracy(retrived[i],tree,1.0)
    temp  = (np.sum(accuracy["get_predicted"] == df['Class'])/len(df))*100
    print('Accuracy : ',temp,end = '')
    print('%')


# def printer(label,k,current):
    # my_list = [label,k,current]
    # print(my_list)

def buildTree(df,count,type1,dict=None):

    if count == 6:
        return dict


    if type1 == 'info':
        label = get_info_gain(df)
    else:
        label = get_variance_impurity(df)

    if label == 'XB':
        count += 1

    if dict is None:
        dict={}
        dict[label] = {}


    for k in [0,1]:
        values =  df[df[label] == k]
        current_leaf1 = set(values['Class'])
        current_leaf = list(current_leaf1)
        if len(counter(values))==1:                                     #Checking purity of subset
            dict[label][k] = current_leaf[0]              #tree[XI][1] =
            #printer(label,k,current_leaf[0])
            #r = current_leaf[0]
            #print('|'+' |'+label+' '+str(k)+':'+str(r))
        else:
             #Calling the function recursively
            #printer([label],[k],buildTree(values))
            #print('|'+label+' '+str(k)+':')
            dict[label][k] = buildTree(values,count,type1)

    return dict


#df1 = pd.read_csv(r"C:\Users\Sagar\Desktop\ML\dataset1\test_set.csv")
#df= pd.read_csv(r"C:\Users\Sagar\Desktop\ML\dataset1\training_set.csv")

input_train = (sys.argv[1])
input_test = (sys.argv[2])
df= pd.read_csv(input_train)
df1 = pd.read_csv(input_test)
t = (buildTree(df,0,sys.argv[3]))



accuracy(df1,t)
