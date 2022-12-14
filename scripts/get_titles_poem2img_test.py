import json

def cleanPrompt(prompt):
    """

    """
    prompt = prompt.replace('\r', '')

    return prompt

def utf8CleanUp(s):
    """

    """
    s_temp = s.encode("UTF-8", "ignore")
    s_temp = s_temp.decode()
    s = cleanPrompt(s_temp)
    return s

def cleanTestPoems(data_poem, datap2i):
    """

    """
    start_i = 3999
    
    for sub_data in datap2i:
        start_i = (sub_data["id"]+1) // 2 + 3998

        title = data_poem[start_i]["title"]

        poem = data_poem[start_i]["poem"]
        new_poem = utf8CleanUp(poem)

        sub_data["poem"] = new_poem
        sub_data["title"] = title
    
    return datap2i




if __name__ == '__main__':

    with open("data/poem/poem.json", "r") as f:
        data_poem = json.load(f)
    with open("data/test_poem2img.json", "r") as f:
        datap2i = json.load(f)
    
    datap2i["poem2img"] = cleanTestPoems(data_poem['poems'], datap2i['poem2img'])

    with open("data/test_poem2img_off.json", "w") as f:
        json.dump(datap2i, f, ensure_ascii=False)