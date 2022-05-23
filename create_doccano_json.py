import re
import json


def createDoccannoJSON(text, predictionResults):
    doccano_item_init = '{ "text": ' + json.dumps(text) + ', "labels": ['
    doccano_item = ""
    doccano_item_finish = "]}"

    for result in predictionResults:
        label = result["label"]
        start_word_index = result["words"][0]["index"]
        end_word_index = result["words"][-1]["index"]

        if label == "LOCATION":
                index_list_locations = find_word_idex(text, start_word_index, end_word_index, "LOCATION")
                doccano_item = populate_doccano_json(doccano_item, index_list_locations)
        elif label == "ACTION":
                index_list_actions = find_word_idex(text, start_word_index, end_word_index, "ACTION")
                doccano_item = populate_doccano_json(doccano_item, index_list_actions)
        elif label == "MODAL":
                index_list_modals = find_word_idex(text, start_word_index, end_word_index, "MODAL")
                doccano_item = populate_doccano_json(doccano_item, index_list_modals)
        elif label == "TRIGGER":
                index_list_triggers = find_word_idex(text, start_word_index, end_word_index, "TRIGGER")
                doccano_item = populate_doccano_json(doccano_item, index_list_triggers)
        elif label == "CONTENT":
                index_list_contents = find_word_idex(text, start_word_index, end_word_index, "CONTENT")
                doccano_item = populate_doccano_json(doccano_item, index_list_contents)

    doccano = doccano_item_init + doccano_item + doccano_item_finish + "\n"

    print(doccano)


def find_word_idex(text, start_word_index, end_word_index, entity):
    index_list = []
    text_words = re.findall( r'\w+|[^\s\w]+', text)

    start_index = 0

    for i in range(start_word_index - 1):
        start_index += len(text_words[i])
        if text[start_index] == " ":
            start_index += 1

    end_index = start_index
    if start_word_index - end_word_index != 0:
        for i in range(start_word_index-1, end_word_index-1):
            print(text_words[i])
            end_index += len(text_words[i])
            if text[end_index] == " ":
                end_index += 1
        end_index += len(text_words[end_word_index - 1])
    else:
        end_index = start_index + len(text_words[end_word_index - 1])

    index_list.append([start_index, end_index, '"' + entity + '"'])
    return index_list


def populate_doccano_json(current, index_list):
    if current != "":
        current += ", "
    for item in index_list:
        converted_list = [str(element) for element in item]
        current += '[' + ','.join(converted_list) + ']'

    return current
