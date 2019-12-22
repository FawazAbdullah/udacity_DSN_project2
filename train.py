# import libraries 
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", type=str, help="data directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint2.pth",
                        help="directory where to save trained model and hyperparameters")
    parser.add_argument("--arch", type=str, default="vgg16",
                        help="Choose architecture from torchvision.models as String")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to train model")
    parser.add_argument("--hidden_units", type=list, default=[700, 300],
                        help="list of hidden layers")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    parser.add_argument("--output", type=int, default=102,
                        help="enter output size")
    
    return parser.parse_args()


def train_test_valid_transformer(train_dir, test_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    
    test_data = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=image_datasets)
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    return train_data, test_data, valid_data


def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader


def isGPU(gpu_arg):
    if gpu_arg:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if device == "cpu":
            print("CUDA was not found on device, using CPU instead.")
            
        return device
        
    return torch.device("cpu")


def init_classifier(model, hidden_nodes = 2096):
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_nodes, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_nodes, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier


def architecture_model(architecture="vgg16"):
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    for param in model.parameters():
        param.requires_grad = False 
    return model


def accuracy_validation(testloader, model, criterion, device):
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            prob = torch.exp(output)
            pred = prob.max(dim=1)

            matches = (pred[1] == labels.data)
            correct += matches.sum().item()
            total += 64

    accuracy = 100*(correct/total)
    return accuracy, test_loss


def train(model, trainloader, testloader, device, criterion, optimizer, epochs = 5, print_every = 30, steps = 0):
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
            
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                        print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
            
                running_loss = 0
                model.train()
            
    print("Done!!")
    return model


def save_checkpoint(model, save_dir, data):
    if not type(save_dir) == type(None):
        if isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            # Save the checkpoint 
            checkpoint = {'classifier': model.classifier,
              'epochs': epochs,
              'optimizer': optimizer,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

            torch.save(checkpoint, 'checkpoint1.pth')
        else:
            print("Directory not found, model will not be saved.")
            
    else:
        print("Directory not found, model will not be saved.")
        
        

# =============================================================================
# Main Function
# =============================================================================

# Function main() is where all the above functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    # Pass transforms in, then create trainloader
    train_data, test_data, valid_data = train_test_valid_transformer(train_dir, test_dir, valid_dir)
    
    trainloader = data_loader(train_data)
    testloader = data_loader(test_data, train=False)
    validloader = data_loader(valid_data, train=False)
    
    # Load Model
    model = architecture_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = init_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = isGPU(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    learning_rate = args.learning_rate
       
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 25
    steps = 0

    # Train the classifier layers using backpropogation
    trained_model = train(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    accuracy_validation(trained_model, testloader, device)
    
    # Save the model
    save_checkpoint(trained_model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()   




