# Import libraries
import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import isGPU
from torchvision import models

def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--topk", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # load the checkpoint
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    test_image = PIL.Image.open(image)

    # Get original dimensions
    width, height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if width < height:
        ratio = height/width
        resize_size=[256, 256**ratio]

    else:
        ratio = width/height
        resize_size=[256**ratio, 256]

    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy
    np_image = np.array(test_image)/255

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std

    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)

    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    """

    # Get Keyword Args for Prediction
    args = arg_parser()

    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)

    # Process Image
    image_tensor = process_image(args.image)

    # Check for GPU
    device = isGPU(gpu_arg=args.gpu);

    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model,
                                                 device, cat_to_name,
                                                 args.topk)

    # Print out the result
    for i in top_flowers:
        print(i)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
