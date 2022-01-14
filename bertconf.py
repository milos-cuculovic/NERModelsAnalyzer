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

def crossval(jsonpath,path):
   data = [json.loads(line) for line in open(jsonpath, 'r')]
   train, test = train_test_split(data, test_size=0.2)
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


def json_jsonbis(jsonPath,jsonpath2):
    ide=0
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION','O']
    dataset=[]
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            listw=sentence.split(" ")
            longS=len(sentence)
            incre=0
            wordd=""
            nertag=[]
            while incre<longS:
                deb=incre
               
                while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
                    wordd=wordd+sentence[incre]
                    removeponct=['"',"-","(",")"]
                    for let in removeponct:
                        if let in wordd:
                            wordd=wordd.replace(let,"")
                    incre=incre+1

                if incre!=deb:
                    label='O'
                    for lab in listLABEL:
                        lab = lab.replace(',', "")
                        if deb >= int(lab.split()[0]):
                            if incre<= int(lab.split()[1]):
                                label=lab.split()[2]
                    nertag.append(LABEL.index(label))
                   
                if incre<longS:
                    if sentence[incre] in string.punctuation:
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
            for word in sentence:
                if word in trigger:
                    label[pos]=2
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

