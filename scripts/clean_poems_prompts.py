import sys
import json
import re

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

def cleanPromptsPoemsNew(data):
    """

    """
    data_nws = data

    for sub_data, sub_data_nws in zip(data, data_nws):
        old_poem = sub_data['poem']
        old_prompt = sub_data['prompt']

        sub_data['poem'] = utf8CleanUp(old_poem)
        sub_data['prompt'] = utf8CleanUp(old_prompt)
        sub_data_nws['poem'] = re.sub(' +', ' ', utf8CleanUp(old_poem))
        sub_data_nws['prompt'] = re.sub(' +', ' ', utf8CleanUp(old_prompt))
    
    return sub_data_nws, data

def cleanPromptsPoemsOld(data_poem, data_p2i):
    """

    """
    data_p2i_nws = data_p2i

    for sub_data, sub_data_nws in zip(data_p2i, data_p2i_nws):
        poem_id = sub_data["id"]
        old_prompt = sub_data["prompt"]
        temp_prompt = old_prompt.split("\"", 1)

        #Fix corrupted img paths
        temp_img_path = sub_data["img_path"].split("/")
        temp_img_path[2] = str(sub_data["id"])
        new_img_path = "/".join(temp_img_path)

        ref_data = data_poem[poem_id - 1]
        ref_poem = ref_data["poem"]

        new_poem = utf8CleanUp(ref_poem)
        if len(temp_prompt) > 1:
            temp_prompt[1] = new_poem + "\""
        else:
            temp_prompt[0] = new_poem + "\""
        new_prompt = " \"".join(temp_prompt)
        
        #Update poem2img.json
        sub_data_nws["img_path"] = new_img_path
        sub_data["poem"] = new_poem
        sub_data["prompt"] = new_prompt

        #Update the nws data
        sub_data_nws["img_path"] = new_img_path
        sub_data_nws["poem"] = re.sub(' +', ' ', new_poem)
        sub_data_nws["prompt"] = re.sub(' +', ' ', new_prompt)

    
    return data_p2i_nws, data_p2i 

def main():
    data = {
        "info": "Stores image location, prompt used to generate, and poem together",
        "poem2img": []
    }

    data_nws = {
        "info": "Stores image location, prompt used to generate, and poem together. NO XTRA WHITE-SPACES",
        "poem2img": []
    }

    with open("data/train_poem2img.json", "r") as f:
        data_old = json.load(f)
    with open("data/train_new_poem2img.json", "r") as f:
        data_new = json.load(f)
    with open("data/poem/poem.json", "r") as f:
        data_poem = json.load(f)
    with open("data/poem2img.json", "r") as f:
        data_p2i = json.load(f)

    data_p2i_old_old_nws = data_p2i
    data_p2i_new_nws = data_new
    data_p2i_old_nws = data_old
    
    data_p2i_old_old_nws["[poem2img"], data_p2i["poem2img"] = cleanPromptsPoemsOld(data_poem["poems"], data_p2i["poem2img"])
    data_p2i_old_nws["poem2img"], data_old["poem2img"] = cleanPromptsPoemsOld(data_poem["poems"], data_old["poem2img"])
    data_p2i_new_nws["poem2img"], data_new["poem2img"] = cleanPromptsPoemsNew(data_new["poem2img"])

    ID = 0

    for norm, nws in zip(data_p2i["poem2img"], data_p2i_old_old_nws["poem2img"]):
        data["poem2img"].append(norm)
        data_nws["poem2img"].append(nws)
        ID = norm["id"]

    for norm, nws in zip(data_old["poem2img"], data_p2i_old_nws["poem2img"]):
        data["poem2img"].append(norm)
        data_nws["poem2img"].append(nws)
        ID = norm["id"]


    for i in range(ID, data_new["poem2img"][0]["id"]-1):
        temp_poem = data_poem["poems"][i]["poem"]
        poem = utf8CleanUp(temp_poem)
        poem_nws = re.sub(' +', ' ', poem)

        sub_data = data_poem["poems"][i]

        for j in range(2):
            obj_p = {
                "id": sub_data['id'],
                "poem" : poem,
                "prompt": poem,
                "img_path": f"data/image/{i+1}/{j}.png",
                "keywords": sub_data['keywords']
            }

            data["poem2img"].append(obj_p)
        
        for j in range(2):
            obj_p_nws = {
                "id": sub_data['id'],
                "poem" : poem_nws, 
                "prompt": poem_nws,
                "img_path": f"data/image/{i+1}/{j}.png",
                "keywords": sub_data['keywords']
            }

            data_nws["poem2img"].append(obj_p_nws)

    for norm, nws in zip(data_new["poem2img"], data_p2i_new_nws["poem2img"]):
        data["poem2img"].append(norm)
        data_nws["poem2img"].append(nws)
        ID = norm["id"]

    with open("data/off_poem2img.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)

    with open("data/off_nws_poem2img.json", "w") as f:
        json.dump(data_nws, f, ensure_ascii=False)

if __name__ == '__main__':
    sys.exit(main())
