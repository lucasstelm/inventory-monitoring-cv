import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Install TorchMetrics
import subprocess
package_name = 'torchmetrics'
subprocess.call(['pip', 'install', package_name])
import torchmetrics

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # To avoid truncated images error

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    
    model.eval()

    # Create a TorchMetrics Precision, Recall, and F1 score object for each class
    num_classes = 5
    precision = torchmetrics.Precision(num_classes=num_classes, task='multiclass', average=None).to(device)
    recall = torchmetrics.Recall(num_classes=num_classes, task='multiclass', average=None).to(device)
    f1 = torchmetrics.F1Score(num_classes=num_classes, task='multiclass', average=None).to(device)

    # Initialize variables for accuracy and loss calculation
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to obtain predicted outputs
            outputs = model(inputs)

            # Compute the precision, recall, and F1 scores for the batch
            precision(outputs, labels)
            recall(outputs, labels)
            f1(outputs, labels)

            # Compute the predicted classes
            _, predicted = torch.max(outputs, dim=1)

            # Update accuracy metrics
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        
    # Get the precision, recall, and F1 scores for each label
    class_precision = precision.compute()
    class_recall = recall.compute()
    class_f1 = f1.compute()

    # Compute the average accuracy and average loss
    avg_accuracy = total_correct / total_samples
    avg_loss = total_loss / len(test_loader)

    # Log the results for each class
    logger.info(f"\nTEST METRICS")
    for class_idx in range(num_classes):
        logger.info(f"Class {class_idx + 1}: Precision={class_precision[class_idx]:.4f}, Recall={class_recall[class_idx]:.4f}, F1={class_f1[class_idx]:.4f}")

    # Log the average accuracy and average loss
    logger.info(f"\nAverage Test Accuracy: {avg_accuracy:.4f}")
    logger.info(f"\nAverage Test Loss: {avg_loss:.4f}")

def train(model, train_loader, validation_loader, epochs, criterion, optimizer, device):

    for epoch in range(epochs):

        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()

        # Initialize variables for accuracy and loss calculation
        total_correct = 0
        total_samples = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass to obtain predicted outputs
                outputs = model(inputs)

                # Compute the predicted classes
                _, predicted = torch.max(outputs, dim=1)

                # Update accuracy metrics
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Compute the loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        # Compute the average accuracy and average loss
        avg_accuracy = total_correct / total_samples
        avg_loss = total_loss / len(validation_loader)

        # Print the average accuracy and average loss
        logger.info(f"\nEPOCH {epoch}")
        logger.info(f"\nAverage Validation Accuracy: {avg_accuracy:.4f}")
        logger.info(f"\nAverage Validation Loss: {avg_loss:.4f}")

        # If last epoch, print the final accuracy and loss
        if epoch == epochs - 1:
            logger.info(f"\nFINAL VALIDATION RESULTS")
            logger.info(f"\nFinal Validation Accuracy: {avg_accuracy:.4f}")
            logger.info(f"\nFinal Validation Loss: {avg_loss:.4f}")

    return model

def net():
    model = models.resnet50(pretrained=False)  # Set pretrained=False to disable transfer learning

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return model

def create_data_loaders(data_dir, batch_size):
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    validation_path = os.path.join(data_dir, 'validation')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validation_loader

def main(args):
    logger.info(f'Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Epochs: {args.epochs}')
    logger.info(f'Input data path: {args.data_dir}, Output model path: {args.model_dir}, Output data path: {args.output_dir}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device {device}")
    
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_dir, args.batch_size)
    model = net()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    logger.info("Starting training")
    model = train(model, train_loader, validation_loader, args.epochs, criterion, optimizer, device)
    
    test(model, test_loader, criterion, device)
    
    logger.info("Saving the model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32, metavar="N", help="Training batch size")
    parser.add_argument('--epochs', type=int, default=2, metavar="N", help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar="LR", help="Learning rate")
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'], help="Training data path in S3")
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'], help="Model output path in S3")
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'], help="Output path in S3")

    args=parser.parse_args()
    
    main(args)