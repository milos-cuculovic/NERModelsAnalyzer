import os
import json
import re

triggers = ['why', 'on the contrary','what','however','either','while','rather','instead of', 'when',
         'in order to','therefore','not only', 'afterwards','once again','or','in order to','in particular',
         'also','if not','if not then','not only','does','albeit','because','is that','that','without','who',
         'whether','is it', 'was it','such as','were they','are they','thus','again','given that','given the',
         'how many','except','nor','both','whose','especialls','for instance','is this','similarly','were there',
         'are there','is there','for the time being','based on','in particular','as currently','perhaps','once',
         'how','otherwise','particularly','overall','although','prior to','At the same time',
         'neither','apart from','besides from','if necessary','hence','how much','by doing so','since','how less'
         'despite','accordingly','etc','always','what kind','unless','which one','if not','if so','even if',
         'not just','not only','besides','after all','generally','similar to','too','like']

ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
path_train_data = os.path.join(ROOT_DIR, 'data_valid_full.json')


def string_found(string1, string2):
   if re.search(r"\b" + re.escape(string1) + r"\b", string2):
      return True
   return False


with open(path_train_data, "r") as read_file:
    replacement = ""
    for line in read_file:
        data = json.loads(line)
        for trigger in triggers:
            if string_found(trigger, data['text']):
                trigger_ok = True
                start_index = data['text'].find(trigger)
                end_index = start_index + len(trigger)
                for label in data['labels']:
                    if start_index == label[0] and end_index == label[1]:
                        trigger_ok = False

                if trigger_ok:
                    line = line.replace("]}",
                                 ", [" + str(start_index) + ", " + str(end_index) + ', "TRIGGER"]]}')

                    print("Found new trigger '" + str(trigger) + " [" + str(start_index) + "," + str(
                        end_index) + "]' in '" + str(data['text']) + "'")

        replacement = replacement + line

    read_file.close()

    write_file = open(path_train_data, "w")
    write_file.write(replacement)
    write_file.close()



