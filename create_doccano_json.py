import re
import json


def createDoccannoJSON(text, predictionResults):
    #punctuation = ['"', "-", "(", ")", ":"]
    punctuation = []
    doccano_item_init = '{ "text": ' + json.dumps(text) + ', "labels": ['
    doccano_item = ""
    doccano_item_finish = "]}"

    results = pipelineoutput_2_doccon(predictionResults, text)
    result_merger = []
    last_label = ""
    for result in results["label"]:
        start_index = result[0]
        end_index = result[1]
        label = result[2]

        if text[start_index:end_index] in punctuation:
            label = last_label

        if last_label != label and last_label != "" and last_label != "O":
            result_merger.append([final_start_index, final_end_index, last_label])

        if last_label != label:
            final_start_index = start_index

        final_end_index = end_index

        last_label = label

    if(len(result_merger) > 1 and len(result_merger) < 8):
        for result in result_merger:
            doccano_item = populate_doccano_json(doccano_item, result)

        doccano = doccano_item_init + doccano_item + doccano_item_finish + "\n"
        return doccano
    else:
        return False


def populate_doccano_json(current, label):
    #   [[33,37,"LOCATION"], [12,22,"ACTION"]]
    if current != "":
        current += ", "

    current += '[' + (str(label[0]) + "," + str(label[1]) + ',"' + str(label[2]) + '"') + ']'

    return current


def pipelineoutput_2_doccon(pipelineoutput, original_text, merge_labels=True):
    label_list = ["O", "B-LOCATION", "I-LOCATION", "B-TRIGGER", "I-TRIGGER",
                  "B-MODAL", "I-MODAL", "B-ACTION", "I-ACTION",  "[CLS]", "[SEP]"]
    words = [item['word'] for item in pipelineoutput]
    labels = [item['entity'] for item in pipelineoutput]
    tags = []
    text = ''
    for idx, (w, l) in enumerate(zip(words, labels)):
        label_pos = int(l[6:]) - 1
        label_name = label_list[label_pos]
        if label_name[1:2] == "-":
            label_name = label_name[2:]
        if w.startswith('##'):
            length = len(text)
            text += w[2:]
            tags.append([length, length + len(w[2:]), label_name, w])
        else:
            if text != '':
                text += ' '
            length = len(text)
            text += w
            tags.append([length, length + len(w), label_name, w])
            length += len(text)
    if not merge_labels:
        return {"text": text, "label": tags}
    else:
        merged_tags = []
        previous_tag = ""
        for idx in range(len(tags)):
            if (tags[idx][1] == len(text) or text[tags[idx][1]] == ' '):
                merged_tags.append(tags[idx])
            else:
                tags[idx + 1][0] = tags[idx][0]
                tags[idx + 1][-1] = tags[idx][-1] + tags[idx + 1][-1].replace('##', '')
            previous_tag = tags[idx][2]
        final_tags = []
        temp_text = original_text.lower()
        accumulated = 0
        for item in merged_tags:
            start = temp_text.find(item[-1])
            end = start + len(item[-1])
            final_tags.append([accumulated + start, accumulated + end, item[-2]])
            temp_text = temp_text[end:]

            accumulated += end

        return {"text": original_text, "label": final_tags}