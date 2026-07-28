import csv
from sklearn.metrics import log_loss, roc_auc_score

# Load predictions (probabilities) from torch_submission.csv
pred_path = 'outputs/torch_submission.csv'
pred_dict = {}
with open(pred_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row['uid']
        prob = float(row['is_pathologic'])
        pred_dict[uid] = prob

# Load validation labels (true labels) from train_labels_JNDlMjr.csv
label_path = 'data/raw/train_labels_JNDlMjr.csv'
true_labels = []
pred_probs = []
with open(label_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row['uid']
        if uid in pred_dict:
            true_labels.append(int(float(row['is_pathologic'])))
            pred_probs.append(pred_dict[uid])

if not true_labels:
    raise ValueError('No overlapping UIDs between predictions and validation labels.')

logloss = log_loss(true_labels, pred_probs)
auroc = roc_auc_score(true_labels, pred_probs)

print(f'Validation Log Loss: {logloss:.4f}')
print(f'Validation AUROC:   {auroc:.4f}')
