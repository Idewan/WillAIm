from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScores():

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def score_poem(self, poem, base_poem, image):
        """
        
        """
        image = Image.open(image)

        inputs = self.processor(text=[poem, base_poem], images=image, return_tensors="pt", padding=True)
        out = self.model(inputs)
        logits_per_image = out.logits_per_image
        probs = logits_per_image.softmax()

        return probs


if __name__ == "__main__":
    import requests

    image = requests.get(url, stream=True).raw

    clip = CLIPScores()
    print(clip.score_poem("a photo of a cat", "a photo of a dog", image))