from PIL import Image
import clip
import torch

class CLIPEval():

    def __init__(self):
        self.device = torch.device('cuda:0')
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def score_poem(self, poem, base_poem, image):
        """
        
        """
        result = {
            "Predicted" : 0,
            "Base" : 0
        }

        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        text = clip.tokenize([poem[:300], base_poem[:300]]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            result["Predicted"] = probs[0][0].item()
            result["Base"] = probs[0][1].item()

        return result
