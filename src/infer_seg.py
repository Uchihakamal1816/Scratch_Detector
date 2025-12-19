import torch, cv2, numpy as np
import segmentation_models_pytorch as smp
from utils import mask_to_bboxes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet("efficientnet-b0", classes=1, in_channels=3)
model.load_state_dict(torch.load("weights/best_unet.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def infer(path, thresh=0.5):
    img0 = cv2.imread(path)
    if img0 is None:
        raise FileNotFoundError(path)

    img0 = img0[:, :, ::-1]
    h0, w0 = img0.shape[:2]

    img = cv2.resize(img0, (512, 512))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.to(DEVICE, dtype=torch.float32)  #error in datatypes

    with torch.no_grad():
        p = torch.sigmoid(model(img))[0, 0].cpu().numpy()

    mask = (p > thresh).astype("uint8")
    mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

    area_ratio = mask.sum() / (h0 * w0)
    bboxes = mask_to_bboxes(mask)
    is_bad = int(mask.sum() > 0)

    return mask, bboxes, area_ratio, is_bad

