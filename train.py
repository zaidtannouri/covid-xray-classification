# train.py
import torch
import torch.optim as optim
import time
from torch import nn

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, model_name="best_model.pth"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode='max')

    def Dtrain(self, num_epochs=10):
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs1, inputs2, labels in self.train_loader:
                inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs1, inputs2)
                labels = labels.long()
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = running_loss / len(self.train_loader)
            avg_train_accuracy = 100 * correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs1, inputs2, labels in self.val_loader:
                    inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs1, inputs2)
                    labels = labels.long()
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{num_epochs}", 
                  f"Train Loss: {avg_train_loss:.4f}", 
                  f"Validation Loss: {avg_val_loss:.4f}", 
                  f"Validation Accuracy: {val_accuracy:.2f}%", 
                  f"Epoch Time: {epoch_time:.2f} seconds")

            self.scheduler.step(val_accuracy)

            # Save model if validation accuracy improves
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')

        print("Training complete!")

        return train_losses, val_losses, train_accuracies, val_accuracies



        
        def train(self, num_epochs=10):
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # Single input model forward pass
                outputs = self.model(inputs)
                labels = labels.long()
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = running_loss / len(self.train_loader)
            avg_train_accuracy = 100 * correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    labels = labels.long()
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{num_epochs}", 
                  f"Train Loss: {avg_train_loss:.4f}", 
                  f"Validation Loss: {avg_val_loss:.4f}", 
                  f"Validation Accuracy: {val_accuracy:.2f}%", 
                  f"Epoch Time: {epoch_time:.2f} seconds")

            self.scheduler.step(val_accuracy)

            # Save model if validation accuracy improves
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), self.model_name)

        print("Training complete!")

        return train_losses, val_losses, train_accuracies, val_accuracies