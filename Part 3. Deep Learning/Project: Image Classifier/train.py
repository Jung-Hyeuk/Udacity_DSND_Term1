####
#### Udacity Data Scientist Nano Degree Program Term 1
#### Part 3. Deep Learning - Project: Image Classifier
#### Hyeuk Jung. 
#### July 31st., 2019
####

##### -------------------- Import statement -------------------- #####
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import numpy as np
from PIL import Image
import json

import time

##### -------------------- Define Functions -------------------- #####
# argument reader
def args_parser():
    parser = argparse.ArgumentParser(description='Use neural network to make prediction on image')
    
    # Set directory to save checkpoints
    parser.add_argument('--save_dir', action = 'store',
                        dest = 'save_directory', default = 'checkpoint.pth',
                        help = 'Enter location to save checkpoint')

    # Choose architecture
    parser.add_argument('--arch', action = 'store',
                        dest = 'pretrained_model', default = 'vgg16',
                        help = 'Enter pretrained model to use (default = VGG16)')
    
    # Set hyperparameters
    parser.add_argument('--learning_rate', action = 'store',
                        dest = 'learning_rate', type = float, default = '0.05',
                        help = 'Enter learning rate for gradient descent (float, default = 0.05)')
    parser.add_argument('--hidden_units', action = 'store',
                        dest = 'units', type = int, default = 500,
                        help = 'Enter hidden units (int, default = 500)')
    parser.add_argument('--epochs', action = 'store',
                        dest = 'epochs', type = int, default = 3,
                        help = 'Enter epochs for training (int, default = 3)')
    
    # Use GPU for training
    parser.add_argument('--gpu', action = "store_true", default = True,
                        help = 'Turn GPU mode on or off (default = Off)')
    
    results = parser.parse_args()
    return results

def data_loader(train_dir, test_dir, valid_dir):
    # Define transforms for the training, testing, and validating data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    # Load data
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    # Loader
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return(train_data, test_data, valid_data, trainloader, testloader, validloader)

def load_pretrained_model(architecture):
    # Loading a pre-trained network (default: VGG16)
    model = getattr(models, architecture)(pretrained = True)
    #exec("model = models.{}(pretrained = True)".format(architecture))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
    
def new_classifier(model, hidden_units):   
    # hidden units: default = 500
    input_units = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units, bias=True)),
        ('relu', nn.ReLU()), 
        ('dropout', nn.Dropout(0.05)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    #model.classifier = classifier
    return classifier

def validation(model, validloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Output (forwarding)
        outputs = model.forward(inputs)
        # Calculate loss
        test_loss += criterion(outputs, labels).item()
        # Calculate accuracy
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return(test_loss, accuracy)

def enable_gpu(args_gpu):
    # Read in argument, if it is set as true, turn on gpu, else use cpu
    if not args_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(device)
    return device

## Training the classifier layers using backpropagation using the pre-trained network to get the features
def trainer(trainloader, validloader, device, optimizer, model, criterion, epochs, steps, print_every):

    running_loss = 0
    for epoch in range(epochs):
        start = time.time()
    
        # Training
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
            
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print( "Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Train loss: {:.4f} | ".format(running_loss / print_every),
                      "Valid loss: {:.4f} | ".format(valid_loss / len(validloader)),
                      "Valid accuracy: {:.4f} | ".format(accuracy / len(validloader)) )
            
            running_loss = 0
            model.train()
            
    end = time.time() - start
    print("Execution time: {} seconds".format(end))

    return(model, optimizer)

# Do validation on the test set
def tester(model, testloader, device):
    correct = 0
    total = 0
    model.to('cuda')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
        
            outputs = model(images)
            _, predicted_outcome = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Number of cases which predictions are correct
            correct += (predicted_outcome == labels).sum().item()

    print("Test accuracy of model: {} %".format(round(100 * correct / total, 4)))
    return

def save_checkpoint(model, train_data, epochs, save_directory):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'num_epochs': epochs}

    return torch.save(checkpoint, save_directory)



##### --------------------- Main statement --------------------- #####
def main():
    
    # Get arguments for training
    args = args_parser()
    save_dir = args.save_directory
    architecture = args.pretrained_model
    learning_rate = args.learning_rate
    hidden_units = args.units
    epochs = args.epochs
    gpu_mode = args.gpu
    
    # Set path for each job
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Get data and loaders
    train_data, test_data, valid_data, trainloader, testloader, validloader = data_loader(train_dir, test_dir, valid_dir)
    
    # Load pretrained model, initial classifier, criterion, and optimizer
    model = load_pretrained_model(architecture)
    model.classifier = new_classifier(model, hidden_units)
    # Only train the classifier parameters, feature parameters are frozen
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # gpu mode check
    device = enable_gpu(gpu_mode)
    model.to(device)
    
    # Train the model
    model, optimizer = trainer(trainloader, validloader, device, optimizer, model, criterion, epochs, steps = 0, print_every = 30)
    
    # Test the model
    tester(model, testloader, device)
    
    # Save the model
    save_checkpoint(model, train_data, epochs, save_dir)
    
##### ---------------------   Run main()   --------------------- #####
if __name__ == '__main__': main()

