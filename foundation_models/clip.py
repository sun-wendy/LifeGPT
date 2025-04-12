import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoProcessor, CLIPModel

class CLIP:
    def __init__(self, clip_model="clip-vit-base-patch32", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = CLIPModel.from_pretrained(f"openai/{clip_model}")
        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.img_mean = torch.tensor(self.processor.image_processor.image_mean, device=self.device).view(1, 1, 3)
        self.img_std = torch.tensor(self.processor.image_processor.image_std, device=self.device).view(1, 1, 3)


    def embed_img(self, img):
        """
        Embeds an image using the CLIP model.
        Parameters:
            img (numpy.ndarray or torch.Tensor): Image of shape (H, W, C) with pixel values in [0, 1].
        Returns:
            torch.Tensor: Normalized image feature vector of shape (D,).
        """
        if not torch.is_tensor(img):
            img = torch.tensor(img, dtype=torch.float32)
            
        H, W, C = img.shape
        
        if H != 224 or W != 224:
            img = rearrange(img, "H W C -> 1 C H W")
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = rearrange(img, "1 C H W -> H W C")
        
        img = img.to(self.device)
        img = (img - self.img_mean) / self.img_std
        img = rearrange(img, "H W C -> 1 C H W")
        with torch.no_grad():
            z_img = self.clip_model.get_image_features(pixel_values=img)
        z_img = z_img / torch.norm(z_img, dim=-1, keepdim=True)
        return z_img.squeeze(0)


    def embed_txt(self, prompts):
        """
        Embeds a list of text prompts using the CLIP model.
        Parameters:
            prompts (list of str): List of strings.
        Returns:
            torch.Tensor: Normalized text feature vectors of shape (B, D).
        """
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            z_text = self.clip_model.get_text_features(**inputs)
        z_text = z_text / torch.norm(z_text, dim=-1, keepdim=True)
        return z_text
