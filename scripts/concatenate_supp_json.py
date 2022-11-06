import json
import sys

def main():

    data = {
        "info": "Stores image location, prompt used to generate, and poem together",
        "poem2img": []
    }

    f1 = open("data/poem/poem.json")
    poem = json.load(f1)['poems']
    f1.close()

    f2 = open("data/poem2img_off.json")
    poem2img = json.load(f2)['poem2img']
    f2.close()

    # f3 = open("data/newpoem2img.json")
    # poem2img2 = json.load(f3)['poem2img']
    # f3.close()


    ID = 0

    for i in poem2img:
        data["poem2img"].append(i)
        ID = i['id']

    ind_ID = ID
    for i in range(ind_ID, len(poem)):
        p = poem[i]
        
        if p['id'] == 4002:
            break
        for j in range(2):
            obj = {
                "id": p['id'],
                "poem" : p["poem"],
                "prompt": p["poem"],
                "img_path": f"data/image/{ID}/{j}.png",
                "keywords": p['keywords']
            }

            data["poem2img"].append(obj)

    # for i in poem2img2:
    #     data["poem2img"].append(i)

    with open('data/poem2img.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    sys.exit(main())