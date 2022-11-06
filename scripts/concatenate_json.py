import json

def main():
    ID = 1
    data = {
        "info": "Free-verse poems extracted from Poetry Foundation.org and split into small sub-poems of length 8ish",
        "poems": []
        }

    f1 = open('data/poem/sub/1.json')
    data1 = json.load(f1)['poems']
    f1.close()

    f2 = open('data/poem/sub/2.json')
    data2 = json.load(f2)['poems']
    f2.close()
    
    f3 = open('data/poem/sub/3.json')
    data3 = json.load(f3)['poems']
    f3.close()
    
    f4 = open('data/poem/sub/4.json')
    data4 = json.load(f4)['poems']
    f4.close()

    for i in data1:
        poem = i
        poem['id'] = ID
        data['poems'].append(poem)
        ID += 1

    for i in data2:
        poem = i
        poem['id'] = ID
        data['poems'].append(poem)
        ID += 1
    
    for i in data3:
        poem = i
        poem['id'] = ID
        data['poems'].append(poem)
        ID += 1
    
    for i in data4:
        poem = i
        poem['id'] = ID
        data['poems'].append(poem)
        ID += 1

    return data
    

if __name__ == '__main__':
    data = main()

    with open('poem.json', 'w') as f:
        json.dump(data, f)