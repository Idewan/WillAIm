from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import torch

class CLIPScores():

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def score_poem(self, poem, base_poem, image):
        """
        
        """
        result = {
            "Predicted" : 0,
            "Base" : 0
        }
        image = Image.open(image)

        inputs = self.processor(text=[poem, base_poem], images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            out = self.model(**inputs)
            logits_per_image = out.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            result["Predicted"] = probs[0][0]
            result["Base"] = probs[0][1]

        return result


if __name__ == "__main__":
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = requests.get(url, stream=True).raw

    clip = CLIPScores()
    print(clip.score_poem("a photo of a cat", "a photo of a dog", image))