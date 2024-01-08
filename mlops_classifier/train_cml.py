import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch

# from mlops_classifier.models.model import MyAwesomeModel

# assume we have a trained model
# model = MyAwesomeModel()
model = torch.load("models/MyAwesomeModel/model.pt")

test_set = torch.load("test_data/test.pt")
test_dataloader = torch.utils.data.DataLoader(test_set, shuffle=False)

preds, target = [], []
for batch in test_dataloader:
    x, y = batch
    probs = model(x)
    preds.append(probs.argmax(dim=-1))
    target.append(y.detach())


target = torch.cat(target, dim=0)
preds = torch.cat(preds, dim=0)
# print(target)
# print(preds)

report = classification_report(target, preds)
with open("classification_report.txt", 'w') as outfile:
    outfile.write(report)
confmat = confusion_matrix(target, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
disp.plot()
plt.savefig('confusion_matrix.png')
