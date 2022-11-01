import jsonlines
import json

with open('./data/DFKI/test.json', 'r') as f:
    text = f.read().strip('\n')
    items = text.split('}\n{')
    for item in items:
        if not item.startswith('{'):
            item = '{' + item
        if not item.endswith('}'):
            item = item + '}'
        ttext = item.replace('null', 'None').replace('\n', ' ').replace(' :', ':')
        ob = eval(ttext)
        print(ob['raw'])