#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:53:51 2021

@author: chams
"""
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
# json=['why','it', 'si', 'on the contrary']
# label="O"
# rest=0
# for y in range(len(json)):
#     for trig in trigger:
#         wordj=json[y]
#         if wordj in trig:
#             if len(wordj)!=len(trig):
#                 triglist=trig.split()
#                 if triglist[0]==wordj:
#                     numword=1
#                     while y+numword<len(json) and numword<len(triglist):
#                         if json[y+numword]==triglist[numword]:
#                             label="trigger"
#                             print("super") 
#                             rest=len(triglist)-1
#                         else:
#                             numword=0
#                             rest=0
#                             break
#                         numword=numword+1
#                 elif rest!=0:
#                     print(wordj)
#                     numm=len(triglist)-rest
#                     print(numm)
#                     if triglist[numm]==wordj:
#                         print(wordj)
#                         label="trigger"
#                         rest=rest-1
#             else:
#                 label="trigger"
word="prout"    
if any(word in x for x in trigger):
    print(word)