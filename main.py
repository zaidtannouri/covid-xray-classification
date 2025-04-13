# main.py
from model import CovidXRayModel, DualCovidXRayModel
from train import Trainer
from dataset import CovidDataset, CovidDatasetDual
from utils import (
    load_covid_dataframe,
    plot_loss_accuracy,
    evaluate_model,
    show_metrics,
    visualize_samples,
    visualize_predictions
)

import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data and labels using utility function
df, label_encoder = load_covid_dataframe("D:\\DL datasets\\covid-ds\\COVID-19_Radiography_Dataset")

# Split DataFrame into train, validation, and test datasets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Labels'], random_state=42)
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG16 normalization
])

# Create datasets using DataFrame splits
train_data = CovidDataset(train_df, transform=transform)
val_data = CovidDataset(val_df, transform=transform)
train_data_dual = CovidDatasetDual(train_df, transform=transform)
val_data_dual = CovidDatasetDual(val_df, transform=transform)

# Visualize sample data from the training set
visualize_samples(train_data.dataframe)

# Initialize models
single_image_model = CovidXRayModel().to(device)
dual_image_model = DualCovidXRayModel(num_classes=4).to(device)  # Assuming 4 classes in your dataset

# Create trainers for both models
trainer_single = Trainer(single_image_model, train_data, val_data, device)
trainer_dual = Trainer(dual_image_model, train_data_dual, val_data_dual, device)

# Train the models
print("Training single image model...")
train_losses_single, val_losses_single, train_accs_single, val_accs_single = trainer_single.train()

print("\nTraining dual image model...")
train_losses_dual, val_losses_dual, train_accs_dual, val_accs_dual = trainer_dual.Dtrain()

# Plot training history for both models
plot_loss_accuracy(train_losses_single, val_losses_single, train_accs_single, val_accs_single)
plot_loss_accuracy(train_losses_dual, val_losses_dual, train_accs_dual, val_accs_dual)

# Evaluate the models on the validation set
print("\nEvaluating single image model...")
true_labels_single, predictions_single = evaluate_model(single_image_model, val_data, 'single_image_model.pth', trainer_single.criterion, device)
print("\nEvaluating dual image model...")
true_labels_dual, predictions_dual = evaluate_model(dual_image_model, val_data_dual, 'dual_image_model.pth', trainer_dual.criterion, device, is_dual_input=True)

# Decode the true labels and predictions
class_names = list(train_data.dataframe['Labels'].unique())  # Class names
decoded_true_single = label_encoder.inverse_transform(true_labels_single)
decoded_preds_single = label_encoder.inverse_transform(predictions_single)
decoded_true_dual = label_encoder.inverse_transform(true_labels_dual)
decoded_preds_dual = label_encoder.inverse_transform(predictions_dual)

# Show metrics for both models
print("\nMetrics for single image model:")
show_metrics(decoded_true_single, decoded_preds_single, class_names)

print("\nMetrics for dual image model:")
show_metrics(decoded_true_dual, decoded_preds_dual, class_names)

# Visualize predictions on test samples (optional)
# visualize_predictions(single_image_model, val_data, class_names, device)
# visualize_predictions(dual_image_model, val_data_dual, class_names, device)
