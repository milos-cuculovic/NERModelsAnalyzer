"""
Created on Fri May 28 08:48:33 2021
@author: chams
"""
import json
import torch
import string
import random
from sklearn.model_selection import train_test_split, KFold

trigger=['why', 'on the contrary','what','however','either','while','rather','instead of', 'when','than',
         'in order to','therefore','not only', 'afterwards','once again','or','in order to','in particular',
         'also','if not','if not then','and','not only','does','albeit','because','is that','that','without','who',
         'whether','is it', 'was it','such as','were they','are they','thus','again','given that','given the',
         'how many','except','nor','both','whose','especialls','for instance','is this','similarly','were there',
         'are there','is there','for the time being','based on','in particular','as currently','perhaps','once',
         'how','otherwise','particularly','overall','although','prior to','At the same time',
         'neither','apart from','besides from','if necessary','hence','how much','by doing so','since','how less'
         'despite','accordingly','etc','always','what kind','unless','which one','if not','if so','even if',
         'not just','not only','besides','after all','generally','similar to','too','like']
# 
def listededict(jsonpath):
    listdict=[]
    with open(jsonpath, 'r') as open_file:
        for jsonObj in open_file:
            data = json.loads(jsonObj)
            listdict.append(data)
    return listdict



def concat(id1, id2, data):
        sample={}
        text_1=data[id1]["text"]
        text_2=data[id2]["text"]
        labels_1=data[id1]["array_agg"]
        labels_2=data[id2]["array_agg"]
        labels=labels_1
        for lab in labels_2:
            exlab=lab.split(",")
            newlist=""
            newlist+=str(int(exlab[0])+len(text_1)+1)+", "
            newlist+=str(int(exlab[1])+len(text_1)+1)+", "
            newlist+=str(exlab[2])
            labels.append(newlist)
        # print(labels)
        sample["text"]=text_1+" "+text_2
        sample["array_agg"]=labels
        return sample

def twoPoint(jsonfile,jsonfile1):
    listdict=listededict(jsonfile)
    sample=[]
    index=0
    concate=False
    for sett in listdict:
        if concate==True:
            sample.append(concat(index-1,index,listdict))
            concate=False
        elif sett["text"][len(sett["text"])-1]==":":
            # print (sett["text"])
            concate=True
        else:
            sample.append(sett)
        index+=1
   
    with open(jsonfile1, 'w') as fp:
        json.dump(sample, fp)
# twoPoint("./final.json","./final1.json")
    

def firstfiltre(jsonpath,jsonpath2,js3):
    json_lines = []
    json_remove=[]
    with open(jsonpath, 'r') as open_file:
        for jsonObj in open_file:
            data = json.loads(jsonObj)
            arrayagg= data['array_agg']
            word=data['text']
            if len(arrayagg)>1 and len(word.split(" "))>3:
                
                json_lines.append(jsonObj)
            if word[len(word)-1]==':' and len(word.split(" "))>3:
                json_remove.append(jsonObj)
    with open(jsonpath2, 'w') as open_file:
        open_file.writelines(''.join(json_lines))
        
# firstfiltre("/home/chams/Documents/mdpi/milosphd/NER_models_reviewer_comments/data-use2.json","/home/chams/Documents/mdpi/milosphd/NER_models_reviewer_comments/data-use3.json","/home/chams/Documents/mdpi/milosphd/NER_models_reviewer_comments/data-use3.json")
def crossval(jsonpath,path):
   data = [json.loads(line) for line in open(jsonpath, 'r')]
   # print(data)
   data=data[0]
   train, test = train_test_split(data, test_size=0.2,train_size=0.8)
   # print(test[0]['text'])
   kf = KFold(n_splits=4, shuffle=True)
   # print(test[0])
   kf.get_n_splits(data)
   i=0
   for train_index, test_index in kf.split(data):
        i+=1
        # print(path+"/test"+str(i)+".json")
        with open(path+"/valid"+str(i)+".json", 'w') as open_test:
           for tri in test_index:
               app_json = json.dumps(data[tri])
               open_test.write(app_json)
               open_test.write('\n')
           open_test.close()

        with open(path+"/train"+str(i)+".json", 'w') as open_train:
            for tra in train_index:
                app_json = json.dumps(data[tra])
                open_train.write(app_json)
                open_train.write('\n')
            open_train.close()


def validdtraincomun(file1,file2):
    with open(file1)as f1:
        lines=f1.readlines()
    f1.close()
    sentencelist=[]
    sentence=""
    for line in lines:
            if line=="\n" :
                sentencelist.append(sentence)
                sentence=""
                print(line)
            else:
                sentence+=line
                   
    with open(file2)as f:
        lines=f.readlines()
    f.close()
    sentencelist2=[]
    sentence=""
    for line in lines:
            if line=="\n" :
                sentencelist2.append(sentence)
                sentence=""
            else:
                sentence+=line
    sim=0
    for sen in sentencelist:
        for sen1 in sentencelist2:
            if sen==sen1:
                sim+=1
                print(sen)
    print(sim)
    print(len(sentencelist))
    print(len(sentencelist2))
    
#validdtraincomun("/home/chams/Downloads/train_1.txt", "/home/chams/Downloads/valid_1.txt")
            
                
                
def convertModel():
    PATH = "/home/chams/Documents/mdpi/pytorch_model.bin"
    torch.save( 'pytorch_model.bin', PATH)

# convertModel()

def removEsc(jsonpath):
    json_lines = []
    with open(jsonpath, 'r') as open_file:
        for line in open_file.readlines():
            if '\\"' in line:
                line=line.replace('\\\\"', "'")
            json_lines.append(line)
    with open(jsonpath, 'w') as open_file:
        open_file.writelines(''.join(json_lines))


def sentenceMean(jsonpath):
    json_lines = []
    with open(jsonpath, 'r') as open_file:
        for line in open_file.readlines():
            if " ACTION" in line:
                if"LOCATION" in line:
                    json_lines.append(line)
    with open(jsonpath, 'w') as open_file:
        open_file.writelines(''.join(json_lines))


# Convert json to conll
def json_conll(jsonPath,conllPath,conlname):
    f = open(conllPath+'/'+conlname,'w')
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            longS=len(sentence)
            incre=0
            wordd=""
            f.write("\n")

            prevlab=""
            while incre<longS:
                deb=incre
                # if sentence[incre] in string.punctuation :
                #     f.write(sentence[incre])
                #     f.write(" ")
                #     f.write("O")
                #     f.write("\n")
                #     incre=incre+1
                # deb=incre
                while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
                    wordd=wordd+sentence[incre]
                    removeponct=['"',"-","(",")"]
                    for let in removeponct:
                        if let in wordd:
                            wordd=wordd.replace(let,"")
                    incre=incre+1

                if incre!=deb:
                    label="O"
                    for lab in listLABEL:
                        lab = lab.replace(',', "")
                        if deb >= int(lab.split()[0]):
                            if incre<= int(lab.split()[1]):
                                if deb>int(lab.split()[0]) and lab.split()[2] in prevlab:
                                    label="I-"
                                else:
                                    label="B-"
                                label=label+lab.split()[2]
                    prevlab=label
                    f.write(wordd)
                    f.write(" ")
                    f.write(label)
                    f.write("\n")

                if incre<longS:
                    if sentence[incre] in string.punctuation:
                        f.write(sentence[incre])
                        f.write(" ")
                        f.write("O")
                        prevlab="O"
                        f.write("\n")
                incre=incre+1
                wordd=""
    f.close()

# remove all label from file
def removeLab(lab,file,file2):
    with open(file) as f:
        lines=f.readlines()
    with open(file2, "w") as f2:
        for line in lines:
            if labelConll(line)!=lab:
                f2.write(line)

def findtrigger(path):
    trigger=[]
    with open(path) as f:
        lines=f.readlines()
    for line in lines:
        lab=labelConll(line)
        if lab=="TRIGGER":
            word=line.partition(' ')[0]
            trigger.append(word.lower())
    res = []
    [res.append(x) for x in trigger if x not in res]
    return res

def findtriggerdict(path):
    trigger={}
    with open(path) as f:
        lines=f.readlines()
    for line in lines:
        lab=labelConll(line)
        if lab=="TRIGGER":
            word=line.partition(' ')[0]
            if word.lower() in trigger:
                nb=trigger.get(word.lower())
                trigger[word.lower()]=nb+1
            else:
               trigger[word.lower()]=1
    return trigger

# remove paragraphe form file never tested NOT SURE IT WORKS
def removeLabParag(lab,file,file2,oneon):
    with open(file) as f:
        lines=f.readlines()
    with open(file2, "w") as f2:
        para=0
        for line in lines:
            if labelConll(line)!=lab:
                f2.write(line)
            elif para==oneon:
                f2.write(line)
            if line.strip()=="":
                if para==oneon:
                    para=1
                else :
                    para=para+1


# Count number of a certain label in a file
def countLab(lb,path):
    with open(path) as f:
        lines = f.readlines()
        lab=0
        for line in lines:
            if labelConll(line)==lb :
                lab=lab+1
        return lab



# Create a list of a paragraph
def listparagr(file,num):
    with open(file) as f:
        parag=[]
        lines =f.readlines()
        lines.append("\n")
        npar=0
        for line in lines:
            if line=="\n" and npar==num:
                return parag
            if line=="\n":
                npar=npar+1
            elif npar ==num:
                parag.append(line)



# UnderSampling by remove label randomly
def randomUndSamp(label,percentage):
    numlab=countLab(label,'/home/chams/Documents/mdpi/train.txt')
    keep=round(numlab*percentage/100)
    with open('/home/chams/Documents/mdpi/train.txt') as f:
        lines = f.readlines()
    while numlab!=keep:
         line = random.choice(lines)
         if labelConll(line)==label :
            deli=line
            with open('/home/chams/Documents/mdpi/train.txt','w') as f:
                for ligne in lines:
                    if ligne.strip() != deli.strip():
                        f.write(ligne)
                    else :
                        deli="000"
            with open('/home/chams/Documents/mdpi/train.txt') as f:
                lines = f.readlines()
         numlab=countLab(label,'/home/chams/Documents/mdpi/train.txt')


def trigConll(file,trigger):
    with open(file)as f:
        lines=f.readlines()
    f.close()
    sentencelist=[]
    wordind=0

    with open(file,'w') as f:
        for line in lines:
                wordind=wordind=wordind+1
                word=""
                if len(line.split())>0:
                    word=line.split()[0]
                else:
                    f.write(line)
                    wordind=0
                    sentencelist=[]
                    continue
                sentencelist.append(word)
                if word in trigger:
                    line= word+" "+"B-TRIGGER\n"
                elif any(word in x for x in trigger):
                    # print(word)
                    for trig in trigger:
                        if word in trig.split():
                            triglist=trig.split()
                            index=triglist.index(word)
                            size=len(triglist)
                            firstword=wordind-index
                            lastword=wordind+size+1
                            if firstword>0 and lastword<len(sentencelist):
                                y=0
                                for i in range(firstword,lastword):
                                    if sentencelist[i]==triglist[y]:
                                        if i==lastword:
                                            if index==0:
                                                line=word+" "+"B-TRIGGER\n"
                                            else:
                                                line=word+" "+"I-TRIGGER\n"
                                    y=y+1
                            triglist=trig.split()
                f.write(line)
        f.close()

def randomOverSamp(label,factor,path):
    numlab=countLab(label,path)
    goal=numlab*factor
    with open (path) as f:
        lines =f.readlines()
    while numlab!= goal:
        print()

# idk en vrai
def numpara(file, nline):
    # numero de paragraphe
    with open(file) as f:
        lines=f.readlines()
        currentline=1
        numparag=1
        while currentline<nline :
            if lines[currentline].strip()=="":
                numparag=numparag+1
            currentline=currentline+1
        return numparag


# Explicit title
def numberofParaginAfile(file):
    parag=0
    with open(file) as f:
        lines=f.readlines()
        for line in lines:
            if line.strip()=="":
               parag=parag+1
    return parag

# return the label last word
def labelConll (line):
    lis = list(line.split(" "))
    length = len(lis)
    lab=lis[length-1]
    return lab.strip()

def json_jsonfile(jsonPath,jsonpath2):
    ide=0
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION','O']
    dataset=[]
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            listw=sentence.split("")
            
def json_jsonbis(jsonPath,jsonpath2):
    ide=0
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION','O']
    dataset=[]
    
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            listw=[]
            longS=len(sentence)
            incre=0
            wordd=""
            nertag=[]
            numword=0
            while incre<longS:
                deb=incre
                while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
                    wordd=wordd+sentence[incre]
                    removeponct=['"',"-","(",")"]
                    for let in removeponct:
                        if let in wordd:
                            wordd=wordd.replace(let,"")
                    incre=incre+1
                
                
                if incre!=deb and wordd!="":
                    label='O'
                    for lab in listLABEL:
                        lab = lab.replace(',', "")
                        if deb >= int(lab.split()[0]):
                            if incre<= int(lab.split()[1]):
                                label=lab.split()[2]
                    listw.append(wordd)   
                    nertag.append(LABEL.index(label))
                   
                if incre<longS:
                    if sentence[incre] in string.punctuation:
                        listw.append(sentence[incre])
                        nertag.append(LABEL.index('O'))

                incre=incre+1
                wordd=""
            datase={"id":ide,"tokens":listw,"ner_tags":nertag}
            ide+=1
            dataset.append(datase)
            
    with open(jsonpath2,'w')as outfile:
        for entry in dataset:
            json.dump(entry, outfile)
            outfile.write('\n')

def tiggerreplacejson(jsonPath):
    dataset=[]
    with open(jsonPath) as fi:
         for jsonObj in fi:
            data = json.loads(jsonObj)
            sentence= data['tokens']
            label= data['ner_tags']
            pos=0
            for i in range(0,len(sentence)):
                word=sentence[i]
                if word in trigger:
                    label[pos]=2
                else:
                    for trig in trigger:
                         if word in trig:
                             if word==trig.split()[0]:
                                 trigbool=True
                                 for j in range (1,len(trig.split())):
                                     if(len(sentence)>i+j):
                                         if sentence[i+j]==trig.split()[j]:
                                             continue
                                         else:
                                           trigbool=False
                                           break
                                     else:
                                           trigbool=False
                                           break 
                                 if trigbool:
                                     for j in range (0,len(trig.split())):
                                         label[pos+j]=2
            datase={"id":data['id'],"tokens":sentence,"ner_tags":label}
            dataset.append(datase)
    with open(jsonPath,'w')as outfile:
        for entry in dataset:
            json.dump(entry, outfile)
            outfile.write('\n')



#fonction qui met tous les paragraphe dans une autre fonction de maniere aleatoire
def shuffleFile(file1,file2,file3):
     with open(file1) as f:
        lines=f.readlines()
        linesrest=[]
        for li in lines:
            linesrest.append(li)
        try:
            while True:
                linesrest.remove('\n')
        except ValueError:
            pass

        with open(file2) as fb:
            lines2=fb.readlines()
        while linesrest:
             nbPar=numberofParaginAfile(file1)
             rand=random.randint(1, nbPar)
             sentencelist=listparagr(file1,rand )
             with open(file2,"w") as f2:
                if lines2:
                    for line2 in lines2:
                        f2.write(line2)
                for sentence in sentencelist:
                    if sentencelist.index(sentence)==0:
                        f2.write("\n")
                    if sentence!="\n" :
                        sentence.replace('\n', '')
                        f2.write(sentence)
             parag=0
             with open(file3,"w") as f1:
                 f1.write("\n")
                 for line in lines:
                    if line.strip()=="":
                        parag=parag+1
                    if parag!= rand:
                        f1.write(line)
                 f1.write("\n")
             linesrest=shuffleFile(file3, file2,file3)
        f.close()
        return linesrest
    

def changeToOther(x,conll):
    f = open(conll, "r+")
    l = f.readlines()
    c = 0
    for i in l:
        if len(i.split())!=0:
            if x in i.split()[-1]:
                replacement=i.split()[0]+" O\n"
                i=replacement
                l[c]=i
        c=c+1     
    with open(conll,'w')as outfile:
        for i in l:       
            outfile.write(i)

