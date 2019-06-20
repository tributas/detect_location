import torch
from PIL import Image
from dataset import Dataset
from model import Model


def infer(path_to_image: str, model):
    image = Image.open(path_to_image)
    image = image.resize((54, 54), Image.ANTIALIAS)
    image = Dataset.preprocess(image)
    with torch.no_grad():
        images = image.unsqueeze(dim=0).cuda()
        prediction = model.eval()(images)
        length_logits = prediction[0].max(1)[1].item()
        pred_digits = ''
        for idx in range(5):
            if str(prediction[idx+1].max(1)[1].item()) != '10':
                pred_digits = pred_digits + str(prediction[idx+1].max(1)[1].item())
    return [int(length_logits), int(pred_digits)]