import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse


#global constants
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

    if h_params["data_augumentation"]:
        train_transform = transforms.Compose([
            transforms.Resize(desired_size),  
             transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor()        
        ])
    else:
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

#This is the class to build the CNN model
class CNN(nn.Module):
    
    #Initialization method
    def __init__(self, h_params):
        super(CNN, self).__init__()
        self.h_params = h_params
        
        #create the no of fileters based on the multiplier
        self.filters =  []
        for i in range(5):
            self.filters.append(int(self.h_params["num_of_filter"]*(self.h_params["filter_multiplier"]**i)))
        print(self.filters)
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = 3
        for i in range(self.h_params["conv_layers"]):
            conv_layer = nn.Conv2d(in_channels, self.filters[i], self.h_params["filter_size"][i])
            self.conv_layers.append(conv_layer)
            if self.h_params["batch_normalization"]:
                self.bn_layers.append(nn.BatchNorm2d(self.filters[i]))
            in_channels = self.filters[i]
        
        #Define Fully connected layers
        f_map_side = self.neurons_in_dense_layer(self.h_params["filter_size"], IMAGE_SIZE)
        self.fc1 = nn.Linear( self.filters[-1]*f_map_side*f_map_side , self.h_params["dense_layer_size"])
        self.fc2 = nn.Linear(self.h_params["dense_layer_size"], NUM_OF_CLASSES)
        self.activation_func = self.get_activation_function(self.h_params["actv_func"])
        # Define dropout layer
        self.dropout = nn.Dropout(p=self.h_params["dropout"])

        
    #forward propogate method
    def forward(self, x):
        for i in range(self.h_params["conv_layers"]):
            x = self.conv_layers[i](x)
            if self.h_params["batch_normalization"]:
                x = self.bn_layers[i](x)
            x = self.activation_func(x)
            x = F.max_pool2d(x, 2, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
    #Returns appropriate activation function
    def get_activation_function(self, activation_func_name):
        if activation_func_name == 'elu':
            return F.elu
        elif activation_func_name == 'gelu':
            return F.gelu
        elif activation_func_name == 'silu':
            return F.silu
        elif activation_func_name == 'selu':
            return F.selu
        elif activation_func_name == 'leaky_relu':
            return F.leaky_relu
    
    #Return the number of neurons that should be in the dense layer
    def neurons_in_dense_layer(self, filter_sizes, image_size):
        for i in range(5):
            image_size = int((image_size - filter_sizes[i] +1)/2)
        return image_size
    

#To evaluate the model on testing data set
def evaluate_testing_model(model,device, loader_data):
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    test_loader =  loader_data["test_loader"]
    with torch.no_grad():  # Disable gradient calculation to speed up computations
      for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
            
          outputs = model(inputs)

          values, predicted = torch.max(outputs, 1)

          correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / loader_data["test_len"]

    print("Test accuracy: ", accuracy)



#To generate the 10*3 grid
def generateGridImage(model, device, loader_data):

    # Initialize lists to store true labels, predicted labels, and images
    class_label_names = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
    true_labels = []
    predicted_labels = []
    images = []

    # Get the first 30 data items from the test dataset
    test_loader = loader_data['test_loader']
    data_iterator = iter(test_loader)
    for i in range(30):
        inputs, labels = next(data_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = model(inputs)

        # Compute predicted labels
        _, predicted = torch.max(outputs, 1)

        # Append true and predicted labels to lists
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # Append images to the list after converting them from tensor to numpy array
        images.extend(inputs.cpu().numpy())

    # Create a grid of images with true and predicted labels
    fig, axs = plt.subplots(10, 3, figsize=(15, 50))

    for i, ax in enumerate(axs.flatten()):
        ax.imshow(np.transpose(images[i], (1, 2, 0)))  # Convert from CHW to HWC format
        ax.axis('off')
        true_label_name = class_label_names[true_labels[i]]
        predicted_label_name = class_label_names[predicted_labels[i]]
        ax.set_title(f'\n\n\nTrue:{true_label_name}\nPredicted:{predicted_label_name}')

    plt.tight_layout()

    # Convert the plot to a wandb image
    wandb_image = wandb.Image(plt)

    # Log the image to wandb
    wandb.log({"Predictions": wandb_image})



#This method will train the model
def train(h_params, training_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(h_params)
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
                print( "epoch  ", epoch, " batch ", i, " accuracy ", crt/labels.shape[0], " loss ", loss.item())

          
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
    
#     Uncomment below line to find the model performance on test data
#     evaluate_testing_model(model,device,training_data)
    
#     Uncomment below line to generate the 10*3 image grid after model is trained
#     generateGridImage(model,device, training_data)

    print('Finished Training')
    PATH = './bestmodel.pth'
    torch.save(model.state_dict(), PATH)

#Parse the commange line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN model with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL proj", help="Specifies the project name used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Sets the number of epochs to train the neural network")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Sets the learning rate used to optimize model parameters")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Specifies the batch size used for training")
    parser.add_argument("-nf", "--num_of_filter", type=int, default=32, help="Sets the number of filters in the convolutional layers")
    parser.add_argument("-fs", "--filter_size", nargs='+', type=int, default=[3, 3, 3, 3, 3], help="Specifies the sizes of filters in the convolutional layers")
    parser.add_argument("-af", "--actv_func", choices=["gelu", "silu", "elu", "leaky_relu"], default="gelu", help="Chooses the activation function for the convolutional layers")
    parser.add_argument("-fm", "--filter_multiplier", type=float, default=2, choices=[0.5, 1, 2], help="Specifies the filter multiplier for the convolutional layers")
    parser.add_argument("-da", "--data_augmentation", action="store_true", help="Specifies whether to use data augmentation or not, default is false, to set true use -da")
    parser.add_argument("-bn", "--batch_normalization", action="store_false", help="Specifies whether to use batch normalization or not, defaut is true, to set to false use -bn")
    parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Sets the dropout rate for the fully connected layers")
    parser.add_argument("-ds", "--dense_layer_size", type=int, default=256, help="Specifies the size of the dense layer")
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
        "num_of_filter": args.num_of_filter,
        "filter_size": args.filter_size,
        "actv_func": args.actv_func,
        "filter_multiplier": args.filter_multiplier,
        "data_augumentation": args.data_augmentation,
        "batch_normalization": args.batch_normalization,
        "dropout": args.dropout,
        "dense_layer_size": args.dense_layer_size,
        "conv_layers": 5,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir
    }


    config = h_params
    training_data = prepare_data(config)
    run = wandb.init(project=args.wandb_project, name=f"{config['actv_func']}_ep_{config['epochs']}_lr_{config['learning_rate']}_init_fltr_cnt_{config['num_of_filter']}_fltr_sz_{config['filter_size']}_fltr_mult_{config['filter_multiplier']}_data_aug_{config['data_augumentation']}_batch_norm_{config['batch_normalization']}_dropout_{config['dropout']}_dense_size_{config['dense_layer_size']}", config=config)
    train(config, training_data) 

main()