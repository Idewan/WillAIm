from PIL import Image
from tqdm import tqdm

import torch
import clip
import json
import pickle
import skimage.io as io
import sys

class PrefixUnit():

    def __init__(self, clip_model_type="ViT-B/32"):
        self.device = torch.device('cuda:0')
        self.clip_model_name = clip_model_type.replace("/", "_")
        self.clip_model, self.preprocess = clip.load(clip_model_type, device=self.device, jit=False)
        self.nfsw_image = io.imread("data/nfsw.png")
    
    def gen_prefix(self, image):
        """
        
        """
        image_pp = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image_pp).cpu()
        
        return prefix

    def save_embeddings(self, embeddings, poems, out_path):
        """
        
        """
        with open(out_path, "wb") as f:
            pickle.dump({
                "clip_embedding" : torch.cat(embeddings, dim=0),
                "poems" : poems}, f)



def main(file_name):
    p_unit = PrefixUnit()

    out_path = f"data/{p_unit.clip_model_name}_train.pkl"

    with open(f'data/{file_name}', 'r') as f:
        data = json.load(f)["poem2img"]
    
    embeddings = []
    poems = []
    count = 0

    for i in tqdm(range(len(data))):
        curr_pair = data[i]
        img_path = curr_pair['img_path']

        image = io.imread(img_path)

        if not (p_unit.nfsw_image == image).all():
            prefix = p_unit.gen_prefix(image=image)

            curr_pair['clip_embedding'] = count

            embeddings.append(prefix)
            poems.append(curr_pair)

            count += 1

        if (i+1) % 500 == 0:
            p_unit.save_embeddings(embeddings, poems, out_path)
    
    p_unit.save_embeddings(embeddings, poems, out_path)
    print(f"We have {count} training examples.")

if __name__ == "__main__":
    file_name = sys.argv[1] #train_nws_poem2img.json or train_poem2img.json
    sys.exit(main(file_name))

        
