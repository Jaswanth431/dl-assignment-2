import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import torchvision.models as models
import argparse


IMAGE_SIZE = 224
NUM_OF_CLASSES = 10


#This method will spilt the training data into training and validation data
def split_dataset_with_class_distribution(dataset, split_ratio):
    train_indices = []
    val_indices = []

    # Hardcoded class ranges based on the provided dataset
    class_ranges = [
        (0, 999),
        (1000, 1999),
        (2000, 2999),
        (3000, 3999),
        (4000, 4998),
        (4999, 5998),
        (5999, 6998),
        (6999, 7998),
        (7999, 8998),
        (8999, 9998)
    ]

    for start, end in class_ranges:
        class_indices = list(range(start, end + 1))
        split_idx = int(len(class_indices) * split_ratio)
        train_indices.extend(class_indices[:split_idx])
        val_indices.extend(class_indices[split_idx:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

#This method will generate train , validation, test data and returns it
def prepare_data(h_params):
    desired_size = (IMAGE_SIZE, IMAGE_SIZE)
    
    train_transform = transforms.Compose([
        transforms.Resize(desired_size),  
        transforms.ToTensor()        
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(desired_size),  
        transforms.ToTensor()        
    ])

    train_data_dir = h_params["train_dir"]
    test_data_dir = h_params["val_dir"]
    train_dataset_total = ImageFolder(train_data_dir, transform=train_transform)
    train_dataset, validation_dataset = split_dataset_with_class_distribution(train_dataset_total, 0.8)

    test_dataset = ImageFolder(test_data_dir, transform=test_transform)
    train_len = len(train_dataset)
    val_len = len(validation_dataset)
    test_len = len(test_dataset)

    batch_size =h_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Return the datasets, loaders, and transforms as a dictionary
    return {
        "train_len": train_len,
        "val_len": val_len,
        "test_len": test_len,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }

#This will load Resnet50 model 
def resnet50Model(h_params):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    k = h_params["last_unfreeze_layers"]
    # Unfreeze the last k layers
    if k > 0:
        for param in list(model.parameters())[-k:]:
            param.requires_grad = True

    return model

#this fucntion will train the model and logs accuracies to wandb
def train(h_params, training_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50Model(h_params)
    model = torch.nn.DataParallel(model, device_ids = [0]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=h_params["learning_rate"])
    train_len = training_data['train_len']
    val_len =  training_data['val_len']
    train_loader = training_data['train_loader']
    val_loader = training_data['val_loader']

    for epoch in range(h_params["epochs"]):
        training_loss = 0.0
        validation_loss = 0.0
        train_correct = 0
        validation_correct = 0
         # Training phase
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            training_loss+=loss.item()
            values, predicted = torch.max(outputs, 1)
            crt = (predicted == labels).sum().item() 
            train_correct += crt
            loss.backward()
            optimizer.step()
            if (i%10 == 0):
                print( "  epoch  ", epoch, " batch ", i, " accuracy ", crt/labels.shape[0], " loss ", loss.item())

          
        # Validation phase
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                values, predicted = torch.max(outputs, 1)
                validation_correct += (predicted == labels).sum().item()
                validation_loss += loss.item()
        
        train_accuracy = train_correct/train_len
        train_loss  = training_loss/len(train_loader)
        validation_accuracy = validation_correct/val_len
        validation_loss = validation_loss/len(val_loader)
        print("epoch: ", epoch, "train accuray:",train_accuracy , "train loss:",train_loss , "val accuracy:", validation_accuracy,"val loss:",validation_loss)
        
        #logging to wandb
        wandb.log({"train_accuracy":train_accuracy, "train_loss":train_loss, "val_accuracy":validation_accuracy, "val_loss":validation_loss, "epoch":epoch})


    print('Finished Training')
    PATH = './model.pth'
    torch.save(model.state_dict(), PATH)

#Parse the commange line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine tune resnet50 model with hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL proj", help="Specifies the project name used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Sets the number of epochs to train the neural network")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Sets the learning rate used to optimize model parameters")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Specifies the batch size used for training")
    parser.add_argument("-lf", "--last_unfreeze_layers", type=int, default=1, help="Sets the number of layer from last to unfreeze")
    parser.add_argument("-td", "--train_dir", type=str, required=True, help="Specifies the folder containing training images")
    parser.add_argument("-vd", "--val_dir", type=str, required=True, help="Specifies the folder containing validation images")

    args = parser.parse_args()

    return args



def main():
    wandb.login()

    args = parse_arguments()
    h_params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "model":"resnet50",
        "last_unfreeze_layers":args.last_unfreeze_layers,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir
    }
    config = h_params
    print(config)
    training_data = prepare_data(config)
    run = wandb.init(project=args.wandb_project, name=f"{config['model']}_ep_{config['epochs']}_bs_{config['batch_size']}_lr_{config['learning_rate']}_last_unfreeze_layers_{config['last_unfreeze_layers']}", config=config)
    train(config, training_data) 
main()