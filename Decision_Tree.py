import math
import time

# Dataset
data = [
['Youth','High','No','Fair','No'],
['Youth','High','No','Excellent','No'],
['Middle','High','No','Fair','Yes'],
['Senior','Medium','No','Fair','Yes'],
['Senior','Low','Yes','Fair','Yes'],
['Senior','Low','Yes','Excellent','No'],
['Middle','Low','Yes','Excellent','Yes'],
['Youth','Medium','No','Fair','No'],
['Youth','Low','Yes','Fair','Yes'],
['Senior','Medium','Yes','Fair','Yes'],
['Youth','Medium','Yes','Excellent','Yes'],
['Middle','Medium','No','Excellent','Yes'],
['Middle','High','Yes','Fair','Yes'],
['Senior','Medium','No','Excellent','No']
]

features = ["Age","Income","Student","Credit"]

train = data[:10]
test = data[10:]


# Entropy
def entropy(rows):
    if len(rows)==0:
        return 0

    yes = 0
    no = 0

    for r in rows:
        if r[-1]=="Yes":
            yes+=1
        else:
            no+=1

    total=len(rows)
    result=0

    for x in [yes,no]:
        if x!=0:
            p=x/total
            result-=p*math.log2(p)

    return result


# Gini Index
def gini(rows):
    if len(rows)==0:
        return 0

    yes=0
    no=0

    for r in rows:
        if r[-1]=="Yes":
            yes+=1
        else:
            no+=1

    total=len(rows)

    return 1-((yes/total)**2+(no/total)**2)


# Split data
def split(rows,column,value):
    result=[]

    for r in rows:
        if r[column]==value:
            result.append(r)

    return result


# Information Gain
def information_gain(rows,column):

    parent=entropy(rows)

    values=set(r[column] for r in rows)

    child=0

    for v in values:
        subset=split(rows,column,v)
        child+=(len(subset)/len(rows))*entropy(subset)

    return parent-child


# ID3 Attribute Selection
def ID3_best(rows):

    gains=[]

    for i in range(len(features)):
        gains.append(information_gain(rows,i))

    return gains.index(max(gains))


# Split Information
def split_info(rows,column):

    values=set(r[column] for r in rows)

    result=0

    for v in values:

        count=0

        for r in rows:
            if r[column]==v:
                count+=1

        p=count/len(rows)

        result-=p*math.log2(p)

    return result


# C4.5 Attribute Selection
def C45_best(rows):

    ratios=[]

    for i in range(len(features)):

        gain=information_gain(rows,i)

        si=split_info(rows,i)

        if si!=0:
            ratios.append(gain/si)
        else:
            ratios.append(0)

    return ratios.index(max(ratios))


# CART Attribute Selection
def CART_best(rows):

    best=999
    attribute=0

    for col in range(len(features)):

        values=set(r[col] for r in rows)

        for v in values:

            left=split(rows,col,v)

            right=[r for r in rows if r[col]!=v]

            if len(left)==0 or len(right)==0:
                continue

            g=(len(left)/len(rows))*gini(left)+(len(right)/len(rows))*gini(right)

            if g<best:
                best=g
                attribute=col

    return attribute


# Build Tree
def build_tree(rows,algorithm):

    if len(rows)==0:
        return "No"

    labels=[r[-1] for r in rows]

    if labels.count(labels[0])==len(labels):
        return labels[0]


    if algorithm=="ID3":
        column=ID3_best(rows)

    elif algorithm=="C4.5":
        column=C45_best(rows)

    else:
        column=CART_best(rows)


    tree={features[column]:{}}

    values=set(r[column] for r in rows)

    for v in values:

        subset=split(rows,column,v)

        tree[features[column]][v]=build_tree(subset,algorithm)

    return tree


# Prediction
def predict(tree,row):

    if type(tree)!=dict:
        return tree

    root=list(tree.keys())[0]

    index=features.index(root)

    value=row[index]

    return predict(tree[root][value],row)


# Metrics
def metrics(actual,predicted):

    TP=FP=FN=0

    for a,p in zip(actual,predicted):

        if a=="Yes" and p=="Yes":
            TP+=1

        elif a=="No" and p=="Yes":
            FP+=1

        elif a=="Yes" and p=="No":
            FN+=1

    precision=TP/(TP+FP) if TP+FP else 0
    recall=TP/(TP+FN) if TP+FN else 0
    f1=(2*precision*recall)/(precision+recall) if precision+recall else 0

    return precision,recall,f1


# Run Algorithms
algorithms=["ID3","C4.5","CART"]

for alg in algorithms:

    start=time.time()

    tree=build_tree(train,alg)

    end=time.time()

    actual=[]
    predicted=[]

    for row in test:

        actual.append(row[-1])
        predicted.append(predict(tree,row))


    correct=0

    for a,p in zip(actual,predicted):

        if a==p:
            correct+=1


    accuracy=(correct/len(test))*100

    precision,recall,f1=metrics(actual,predicted)


    print("\nAlgorithm:",alg)
    print("Tree:",tree)
    print("Accuracy:",accuracy,"%")
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 Score:",f1)
    print("Training Time:",end-start)