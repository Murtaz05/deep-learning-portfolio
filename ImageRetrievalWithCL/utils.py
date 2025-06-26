import torch
from sklearn.metrics import classification_report



def accuracy(true_label, pred_label):
    correct_pred = torch.sum(true_label == pred_label).float()
    return round(correct_pred.item() / len(true_label), 3)

def precision(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).float()
    fp = torch.sum((true_label == 0) & (pred_label == 1)).float()
    return round((tp / (tp + fp)).item(), 3) if (tp + fp) > 0 else 0

def recall(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).float()
    fn = torch.sum((true_label == 1) & (pred_label == 0)).float()
    return round((tp / (tp + fn)).item(), 3) if (tp + fn) > 0 else 0

def f1_score(true_label, pred_label):
    prec = precision(true_label, pred_label)
    rec = recall(true_label, pred_label)
    return round(2 * (prec * rec) / (prec + rec), 3) if (prec + rec) > 0 else 0

def confusion_matrix(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).item()
    tn = torch.sum((true_label == 0) & (pred_label == 0)).item()
    fp = torch.sum((true_label == 0) & (pred_label == 1)).item()
    fn = torch.sum((true_label == 1) & (pred_label == 0)).item()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def classification_rep(true_label, pred_label):
    return classification_report(true_label.cpu().numpy(), pred_label.cpu().numpy())

def metrics(true_label, pred_label):
    acc = accuracy(true_label, pred_label)
    prec = precision(true_label, pred_label)
    rec = recall(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    cm = confusion_matrix(true_label, pred_label)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": cm
    }



def contrastive_loss(d, y, alpha=1.0):
    loss = torch.mean(y * d.pow(2) +  (1 - y) * torch.clamp(alpha - d, 0).pow(2))
    return loss


def predict_label(d, threshold=0.52): #because mean distance is 0.5142
    return (d < threshold).float()