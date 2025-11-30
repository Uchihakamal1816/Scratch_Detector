import os, cv2
from sklearn.metrics import precision_score, recall_score
from infer_seg import infer

def load_names(path):
    return [x.strip() for x in open(path).readlines()]

names = load_names("data/splits/test.txt")

y_true = []
y_pred = []

for name in names:
    mask_path = "data/masks/" + name
    gt = 1 if os.path.exists(mask_path) else 0
    pred_mask, _, _, pred = infer("data/images/" + name)

    y_true.append(gt)
    y_pred.append(pred)

from sklearn.metrics import confusion_matrix

print("Recall (bad images) =", recall_score(y_true, y_pred, pos_label=1))
print("Precision (bad predictions) =", precision_score(y_true, y_pred, pos_label=1))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
