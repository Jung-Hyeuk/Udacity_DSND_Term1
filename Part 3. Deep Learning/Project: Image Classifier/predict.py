####
#### Udacity Data Scientist Nano Degree Program Term 1
#### Part 3. Deep Learning - Project: Image Classifier
#### Hyeuk Jung. 
#### July 31st., 2019
####

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
import matplotlib.pyplot as plt

import time

def args_parser():
    parser = argparse.ArgumentParser(description='Use neural network to make prediction on image')
    
    # Set image path (test image)
    parser.add_argument('--iamge_path', action = 'store',
                        dest = 'imagepath', default = '../../../data/flowers/test/20/image_04912.jpg',
                        help = 'Enter path to images') #/aipnd-project
    
    # Set directory to load checkpoints
    parser.add_argument('--save_dir', action = 'store',
                        dest = 'save_directory', default = 'checkpoint.pth',
                        help = 'Enter location of the checkpoint')
    
    # Choose architecture
    parser.add_argument('--arch', action = 'store',
                        dest = 'pretrained_model', default = 'vgg16',
                        help = 'Enter pretrained model to use (default = VGG16)')
    
    # Set top_k most likely classes
    parser.add_argument('--top_k', action = 'store',
                        dest = 'topk', type = int, default = 5,
                        help = 'Enter number of top most likely classes to view (int, default = 5)')

    # Mapping of categories to real names
    parser.add_argument('--category_names', action = 'store',
                        dest = 'category_names', default = 'cat_to_name.json',
                        help = 'Enter a mapping category to image') #cat_name_dir
    
     # Use GPU for training
    parser.add_argument('--gpu', action = "store_true", default = True,
                        help = 'Turn GPU mode on or off (default = Off)')
    
    results = parser.parse_args()

    return results

def load_pretrained_model(architecture):
    # Loading a pre-trained network (default: VGG16)
    model = getattr(models, architecture)(pretrained = True)
    #exec("model = models.{}(pretrained = True)".format(architecture))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
    
def load_checkpoint(model, save_directory):
    checkpoint = torch.load(save_directory)
    
    #model = models.vgg16(pretrained=True)   
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    # Resize -> crop out the center 224x224
    # pil_image_w, pil_image_h = pil_image.size
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    # Transform the image size for a PyTorch model
    resized_image = transform(pil_image)
    # Convert into numpy array
    np_image = np.array(resized_image)
    
    return(np_image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
  
def enable_gpu(args_gpu):
    # Read in argument, if it is set as true, turn on gpu, else use cpu
    if not args_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(device)
    return device

def predict(loaded_model, checkpoint, image_path, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    # Loading a model and image
    # Rebuild the model by loading a checkpoint 
    #loaded_model = load_checkpoint(model, checkpoint).to(device)
    image_array = process_image(image_path)
    image_tensor = torch.from_numpy( np.expand_dims(image_array, axis = 0) ).float().to(device) 
    
    # Test the network
    loaded_model.eval()
    with torch.no_grad(): 
        output = loaded_model.forward(image_tensor)
        
    # Calculate the class probabilities (softmax) for img
    probs = torch.exp(output)
    top_probs = probs.topk(topk)[0]
    top_labels = probs.topk(topk)[1]
    
    # Get top k's information
    top_probs_list = np.array(top_probs)[0]
    top_labels_list = np.array(top_labels)[0]
    
    # Get index and class information from the loaded model
    class_to_idx = loaded_model.class_to_idx
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    
    # Get class name for top k using its index number
    top_labels_list_class = []
    for labels in top_labels_list:
        top_labels_list_class.append(idx_to_class[labels])
    
    #print(top_labels_list_class)
    return(top_probs_list, top_labels_list_class)


##### --------------------- Main statement --------------------- #####
def main():
    
    # Get arguments for training
    args = args_parser()
    
    imagepath = args.imagepath
    save_directory = args.save_directory
    architecture = args.pretrained_model
    topk = args.topk
    category_names = args.category_names
    gpu_mode = args.gpu
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load pretrained model, initial classifier, criterion, and optimizer
    model = load_pretrained_model(architecture)
    print(model)
    # Rebuild the model by loading a checkpoint 
    loaded_model = load_checkpoint(model, save_directory)
    print(loaded_model)
    
    # gpu mode check
    device = enable_gpu(gpu_mode)
    model.to(device)
    
    # Predict top k most likely classes
    probs, classes = predict(loaded_model, save_directory, imagepath, topk, device)
    print(probs)
    print(classes)
    
    # Plot results 1. Plot test image
    #image = process_image(imagepath)
    #fig, ax = plt.subplots(2, figsize = (5, 10))
    flower_name = cat_to_name[ imagepath.split('/')[6] ]
    print(flower_name)
    #imshow(image, ax = ax[0], title = flower_name) #
    #ax[0].set_title(flower_name)

    # Plot results 2. Plot probabilities from prediction for the test image
    classes_name = []
    for i in classes:
        classes_name.append( cat_to_name[i] )
    print(classes_name)
    
    #ax[1].barh(y = classes, width = probs)
    #ax[1].set_yticks(classes)
    #ax[1].set_yticklabels(classes_name)
    #ax[1].invert_yaxis()
    #plt.show()
    print("Flower is most likely to be {}, with a probability of {}%".format(classes_name[0], round(probs[0]*100, 4)))
    
    
##### ---------------------   Run main()   --------------------- #####
if __name__ == '__main__': main()


