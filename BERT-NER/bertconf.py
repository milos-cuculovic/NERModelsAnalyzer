"""
Created on Fri May 28 08:48:33 2021
@author: chams
"""
import json
import string 
import random
import os

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
         
         
def removEsc(jsonpath):
    json_lines = []
    with open(jsonpath, 'r') as open_file:
        for line in open_file.readlines():
            if '\\"' in line:
                line=line.replace('\\\\"', "'")
            json_lines.append(line)
    with open(jsonpath, 'w') as open_file:
        open_file.writelines(''.join(json_lines))
        
        
def setenceMean(jsonpath):
    json_lines = []
    with open(jsonpath, 'r') as open_file:
        for line in open_file.readlines():
            if " ACTION" in line:
                if"LOCATION" in line:
                    json_lines.append(line)
    with open(jsonpath, 'w') as open_file:
        open_file.writelines(''.join(json_lines))
        
# Convert json to conll
# def json_conllTRIG(jsonPath,conllPath,trigger):
#     f = open(conllPath+'/conll.txt','w')
#     with open(jsonPath) as fi:
#         for jsonObj in fi:
#             data = json.loads(jsonObj)
#             listLABEL= data['array_agg']
#             sentence= data['text']
#             longS=len(sentence)
#             incre=0
#             wordd=""
#             f.write("\n")
#             while incre<longS:
#                 deb=incre
#                 if sentence[incre] in string.punctuation:
#                     f.write(sentence[incre])
#                     f.write(" ")
#                     f.write("O")
#                     f.write("\n")
#                     incre=incre+1
#                 deb=incre
#                 while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
#                     wordd=wordd+sentence[incre]   
#                     incre=incre+1
    
#                 if incre!=deb:
#                     label="O"
#                     for lab in listLABEL:  
#                         lab = lab.replace(',', "") 
#                         if deb >= int(lab.split()[0]):
#                             if incre<= int(lab.split()[1]):
#                                 label=lab.split()[2]
#                     f.write(wordd)
#                     f.write(" ")
#                     # if label=="CONTENT":
#                     #     label="O"
#                     if wordd.lower() in trigger:
#                         label="TRIGGER"
#                     f.write(label)
#                     f.write("\n")
               
#                 if incre<longS:
#                     if sentence[incre] in string.punctuation:
#                         f.write(sentence[incre])
#                         f.write(" ")
#                         f.write("O")
#                         f.write("\n")
#                 incre=incre+1
#                 wordd=""
#     f.close()


# Convert json to conll
# def json_conllTRIGsent(jsonPath,conllPath,trigger):
#     f = open(conllPath+'/conlltest.txt','w')
#     with open(jsonPath) as fi:
#         for jsonObj in fi:
#             data = json.loads(jsonObj)
#             listLABEL= data['array_agg']
#             sentence= data['text']
#             sentencelist=sentence.split()
#             rest=0
#             longS=len(sentence)
#             incre=0
#             wordd=""
#             y=0
#             f.write("\n")
#             while incre<longS:
#                 deb=incre
#                 if sentence[incre] in string.punctuation:
#                     f.write(sentence[incre])
#                     f.write(" ")
#                     f.write("O")
#                     f.write("\n")
#                     incre=incre+1
#                     rest=0
#                 deb=incre
#                 while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
#                     wordd=wordd+sentence[incre]   
#                     incre=incre+1
    
#                 if incre!=deb:
#                     label="O"
#                     for lab in listLABEL:  
#                         lab = lab.replace(',', "") 
#                         if deb >= int(lab.split()[0]):
#                             if incre<= int(lab.split()[1]):
#                                 label=lab.split()[2]
#                     f.write(wordd)
#                     f.write(" ")
                    
                    
                    
                    
                    
#                     for trig in trigger:
#                         wordj=wordd.lower()
#                         if wordj in trig:
#                             print(sentencelist)
#                             print(wordj)
#                             if len(wordj)!=len(trig) and len(trig.split())>1:
#                                 triglist=trig.split()
#                                 if triglist[0]==wordj:
#                                     numword=1
#                                     while y+numword<len(sentencelist) and numword<len(triglist):
#                                         if sentencelist[y+numword]==triglist[numword]:
#                                             label="trigger"
#                                             rest=len(triglist)-1
#                                         else:
#                                             numword=0
#                                             rest=0
#                                             break
#                                         numword=numword+1                    
#                                 elif rest!=0:
#                                     numm=len(triglist)-rest
#                                     print('num=')
#                                     print(rest)
#                                     print(len(triglist))
#                                     print(triglist)
#                                     if triglist[numm]==wordj:
#                                         print(wordj)
#                                         label="trigger"
#                                         rest=rest-1
#                             else:
#                                 label="trigger"
                                
                                
                                
                                
  
                                
                                
                                
                                
                                
                                
#                     f.write(label)
#                     f.write("\n")
#                 if incre<longS:
#                     if sentence[incre] in string.punctuation:
#                         y=y-1
#                         f.write(sentence[incre])
#                         f.write(" ")
#                         f.write("O")
#                         f.write("\n")
#                 incre=incre+1
#                 wordd=""
#                 y=y+1
#     f.close()

# Convert json to conll
def json_conll(jsonPath,conllPath):
    f = open(conllPath+'/conlls.txt','w')
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            longS=len(sentence)
            incre=0
            wordd=""
            f.write("\n")
            while incre<longS:
                deb=incre
                if sentence[incre] in string.punctuation:
                    f.write(sentence[incre])
                    f.write(" ")
                    f.write("O")
                    f.write("\n")
                    incre=incre+1
                deb=incre
                while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
                    wordd=wordd+sentence[incre]   
                    incre=incre+1
    
                if incre!=deb:
                    label="O"
                    for lab in listLABEL:  
                        lab = lab.replace(',', "") 
                        if deb >= int(lab.split()[0]):
                            if incre<= int(lab.split()[1]):
                                label=lab.split()[2]
                    f.write(wordd)
                    f.write(" ")
                    # if label=="CONTENT":
                    #     label="O"
                    f.write(label)
                    f.write("\n")
               
                if incre<longS:
                    if sentence[incre] in string.punctuation:
                        f.write(sentence[incre])
                        f.write(" ")
                        f.write("O")
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
    
    with open(file+"1",'w') as f:  
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
                    line= word+" "+"TRIGGER\n"
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
                                           line=word+" "+"TRIGGER\n"
                                    y=y+1
                            triglist=trig.split()
                f.write(line)
        f.close()
            
            
# Stoped for Milos idea 
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
                    # try:
                    #    while True:
                    #         lines2.remove('\n')
                    # except ValueError:
                    #     pass
                    
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
    

# randomUndSamp("O", 10)     

# trigger=findtriggerdict("/home/chams/Documents/mdpi/conlls.txt")
# print (trigger)
trigConll("/home/chams/Documents/mdpi/conlls.txt", trigger)
# json_conllTRIGsent("/home/chams/Documents/mdpi/data1.json","/home/chams/Documents/mdpi", trigger)    
# # removeLab("O","/home/chams/Documents/mdpi/conll.txt","/home/chams/Documents/mdpi/conll2.txt")    
# json_conll("/home/chams/Documents/mdpi/data1.json","/home/chams/Documents/mdpi")    
# removeLabParag("O","/home/chams/Documents/mdpi/conll.txt","/home/chams/Documents/mdpi/conllshuffle.txt",3) 
# # shuffleFile('/home/chams/Documents/mdpi/train.txt', '/home/chams/Documents/mdpi/conllshuffle.txt','/home/chams/Documents/mdpi/datasetxt3.txt')
# # # # # os.remove("/home/chams/Documents/mdpi/datasetxt3.txt")      
# print("TRIG")
# print(countLab("TRIGGER","/home/chams/Documents/mdpi/conll.txt"))        
# print("O")
# print(countLab("O","/home/chams/Documents/mdpi/conll.txt"))             
# print("loc")
# print(countLab('LOCATION',"/home/chams/Documents/mdpi/conll.txt"))       
# print("modal")
# print(countLab("MODAL","/home/chams/Documents/mdpi/conll.txt"))       
# print("action")
# print(countLab("ACTION","/home/chams/Documents/mdpi/conll.txt"))       
# print("content")
# print(countLab('CONTENT',"/home/chams/Documents/mdpi/conll.txt"))
# # # print(findtrigger("/home/chams/Documents/mdpi/conll.txt"))
# setenceMean("/home/chams/Documents/mdpi/data1.json")
# removEsc("/home/chams/Documents/mdpi/milosdat.json")
# print(numberofParaginAfile("/home/chams/Documents/mdpi/nomean/valid.txt"))