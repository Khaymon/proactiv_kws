import torchvision
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

from model import Spotter
from data_loader import ImageDatasetFromPt
from trainable_tensor import TrainableTensor
from utils import save_data
from params import get_params

'''
This Script contains the main training script for the input optimization.
Author: @vidyaprsd
'''

def input_optimize(model, data_loader, params):
    #code to optimize the inputs

    # iterate over all images changed by $transform
    for index, batch in enumerate(data_loader):
        print(index)

        outputs         = model(batch['input'])
        loss            = params["criterion"](outputs, batch['target'])

        save_data(log_subdir, index, batch['input'], outputs, batch['target'], 'transform')

        # initialize a trainable input tensor with the input images, i.e., batch['input']
        trainable_input = TrainableTensor(batch['input'].clone())

        # Initialize the optimizer parameters with the parameters of the trainable input tensor.
        # Usually in model training, the optimizer parameters is set to model.parameters(). Here,
        # We change that to trainable_input.parameters() and to update the input while keeping the model constant.
        optimizer       = torch.optim.Adam(trainable_input.parameters(), lr=params["learning_rate"])
        prev_loss       = 9999

        for epoch in range(params["num_iters"]):
            # The trainable input is passed as a model input and the corresponding loss computed.
            outputs     = model(trainable_input.tensor)
            _, predict  = torch.max(outputs.data, 1)
            loss        = params["criterion"](outputs, batch['target'])

            optimizer.zero_grad()  # Reset gradients to 0
            loss.backward()  # Recompute gradients via backpropagation w.r.t the trainable_input parameters, i.e., input image pixels.
            optimizer.step()  # Update the trainable_input parameters/pixels.
            prev_loss   = loss

            if loss < params["cutoff_loss"]:  # break if the minimum loss is reached.
                break

            if epoch % 50 == 0 or epoch == params["num_iters"] - 1:
                print("Batch: ", index, "; Epoch: ", epoch, "; Loss: ", np.round(loss.item(), 4))

        outputs         = model(trainable_input.tensor)
        save_data(log_subdir, index, trainable_input.tensor, outputs, batch['target'], 'projected')



if __name__ == '__main__':
    params              = get_params() #set the traing params in params.py
    device              = 'cuda' if params["cuda"] else 'cpu'

    #Normalize inputs
    preproc             = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,),
                                                                                                           (0.3081,))])

    #load pretrained model. We do not change the weights of these.
    model               = Spotter(40, 512, 256, 3).to(device)
    checkpoint          = torch.load(params["pretrained_model_path"], map_location=lambda storage,
                                                                                                          loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #iterate over transforms T
    for transform in params["transforms"]['values']:
        print(transform)

        log_subdir      = os.path.join(params["logdir"], '_'.join(map(str, transform)))
        if not os.path.exists(log_subdir):
            os.makedirs(log_subdir)

        #initialize the dataset loader params to generate images changed by $transform value
        dataset_params  = {'data_path': params["data_path"], 'preproc': preproc,
                           'cnt': int(params["batch_size_test"] / params["numclasses"]),
                           'cuda':  params["cuda"]}

        for i in range(len(transform)):
            dataset_params[params["transforms"]['types'][i]] = transform[i]

        dataset         = ImageDatasetFromPt(**dataset_params)
        data_loader     = DataLoader(dataset=dataset, batch_size=params["batch_size_test"], shuffle=False,
                                     num_workers=0, pin_memory=False)

        #optimize all instances transformed by T
        input_optimize(model, data_loader, params)
