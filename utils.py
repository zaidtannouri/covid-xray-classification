import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import torch
import torch.nn.functional as F
from pathlib import Path
import os


def load_covid_dataframe(image_dir_path):
    image_dir = Path(image_dir_path)

    filepaths_images = list(image_dir.glob(r'**/images/*.png'))
    filepaths_masks = list(image_dir.glob(r'**/masks/*.png'))
    labels = list(map(lambda x: os.path.split(x)[0].split('\\')[-2], filepaths_images))

    filepaths_images = pd.Series(filepaths_images, name='Filepath_images').astype(str)
    filepaths_masks = pd.Series(filepaths_masks, name='Filepath_masks').astype(str)
    labels = pd.Series(labels, name='Labels')

    images = pd.concat([filepaths_images, filepaths_masks, labels], axis=1)

    label_encoder = LabelEncoder()
    images['Labels'] = label_encoder.fit_transform(images['Labels'])

    return images, label_encoder

def visualize_samples(df, n_samples=5):
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3))
    for j in range(n_samples):
        for idx, i in enumerate(['Filepath_images', 'Filepath_masks']):
            img = Image.open(df.iloc[j][i])
            axes[j, idx].imshow(img.convert("RGB"))
            axes[j, idx].set_title(f"{i} - Label: {df.iloc[j]['Labels']}")
            axes[j, idx].axis('off')
    plt.tight_layout()
    plt.show()

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_precision_recall(y_true, y_scores, class_names):
    plt.figure(figsize=(10, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, criterion, device, is_dual_input=False):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            if is_dual_input:
                inputs1, inputs2, labels = [x.to(device) for x in batch]
                outputs = model(inputs1, inputs2)
            else:
                inputs, labels = [x.to(device) for x in batch]
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n✅ Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"✅ Test Accuracy: {100 * correct / total:.2f}%\n")

    return all_labels, all_preds

def show_metrics(y_true, y_pred, class_names=None):
    print("📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))
    plot_confusion_matrix(y_true, y_pred, class_names)

def visualize_predictions(model, data_loader, class_names, device, is_dual_input=False, num_samples=6):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for batch in data_loader:
            if is_dual_input:
                inputs1, inputs2, labels = [x.to(device) for x in batch]
                outputs = model(inputs1, inputs2)
                imgs = inputs1
            else:
                imgs, labels = [x.to(device) for x in batch]
                outputs = model(imgs)

            preds = outputs.argmax(dim=1)

            for i in range(len(imgs)):
                if images_shown >= num_samples:
                    break
                plt.subplot(2, (num_samples + 1) // 2, images_shown + 1)
                img = imgs[i].cpu().permute(1, 2, 0).numpy()
                plt.imshow(img)
                plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
                plt.axis("off")
                images_shown += 1
            if images_shown >= num_samples:
                break

    plt.tight_layout()
    plt.show()




