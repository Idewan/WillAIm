import sys
import json
import re
from collections import Counter

def cleanPoem(poem):
    poem = poem.replace('\r', '')
    poem = poem.replace('\n', '')
    poem = re.sub(' +', ' ', poem.lower())
    return poem.split(" ")

def get_average_words(data):
    n = len(data)
    sum_ = 0

    for sub_data in data:
        sum_ += len(cleanPoem(sub_data["poem"]))

    return sum_ / n

def get_lines(data):
    n = len(data)
    sum_ = 0
    
    for sub_data in data:
        sum_ += sub_data["poem"].count("\n")
    
    return sum_ / n

def get_unique_words(data):
    c_poem = cleanPoem(data[0]["poem"])
    counter = Counter(c_poem)

    n = len(data)

    for i in range(1, len(data)):
        c_poem = cleanPoem(data[i]["poem"])
        c_counter = Counter(c_poem)
        counter.update(c_counter)

    return len(counter)


def main():
    with open("data/poem/poem.json", "r") as f:
        total = json.load(f)

    with open("data/train_poem2img.json", "r") as f:
        train = json.load(f)

    with open("data/test_poem2img.json", "r") as f:
        test = json.load(f)


    print("== TOTAL ==")
    print(f"There are {len(total['poems'])} poems")
    print(f"There are # unique words: {get_unique_words(total['poems'])}")
    print(f"There are {get_average_words(total['poems'])} number of words on average")
    print(f"There are an average # lines: {get_lines(total['poems'])}")
    
    print("== TRAIN ==")
    print(f"There are {len(train['poem2img']) // 2} poems")
    print(f"There are {len(train['poem2img'])} images")
    print(f"There are # unique words: {get_unique_words(train['poem2img'])}")
    print(f"There are {get_average_words(train['poem2img'])} number of words on average")
    print(f"There are an average # lines: {get_lines(train['poem2img'])}")


    print("== TEST ==")
    print(f"There are {len(test['poem2img'])} poems")
    print(f"There are {len(test['poem2img'])} images")
    print(f"There are # unique words: {get_unique_words(test['poem2img'])}")
    print(f"There are {get_average_words(test['poem2img'])} number of words on average")
    print(f"There are an average # lines: {get_lines(test['poem2img'])}")


if __name__ == "__main__":
    sys.exit(main())