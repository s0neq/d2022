# -*- coding: utf-8 -*-

import os
import logging
from logging.handlers import RotatingFileHandler
import torch
from torchvision import transforms
import cv2
from PIL import Image

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


from rudalle.pipelines import generate_images, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_ruclip
from rudalle.utils import seed_everything

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


ERRNAME, TRUENAME = "lem_sent", "lem_sent_true"
device = "cuda:0"

def calculate_fid(act1, act2):
    """
    calculate frechet inception distance (FID)
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_imgs(name):
    imgs_list = []
    path = "./generated/"+name
    generated = os.listdir(path)
    for filename in generated:
        imgs_list.append(Image.open(path+"/"+filename))
    return imgs_list


def trans_img(imgs_list):
    tensor_list = []
    for img in imgs_list :
        # add white padding to the images to reach the expected size
        width, height = img.size[1], img.size[0]
        pad = int((299-width)/2)

        if (pad > 0):
            img = cv2.copyMakeBorder(np.array(img), pad, pad, pad, pad, borderType = cv2.BORDER_CONSTANT, value = (255,255,255))

        img = cv2.resize(np.array(img),(299,299))

        # convert the image to tensor and apply normalization
        tensor_img = toTens(img)  
        norm_tensor = normalize(tensor_img) 

        # create batch
        batch_tensor = torch.unsqueeze(norm_tensor, 0)  # add extra dimension for batch value
        if is_cuda:
            batch_tensor = batch_tensor.to(device)

        tensor_list.append(batch_tensor)
    return tensor_list


def get_distr(tensor_list, feat_inception):
    res = []
    for i in range(len(tensor_list)):
        output_feat = feat_inception(tensor_list[i])
        vec_feat = output_feat['flatten'].cpu().detach().numpy().flatten()
        res.append(vec_feat)
    return np.stack(res)


is_cuda = torch.cuda.is_available()

imgs_list = get_imgs(ERRNAME)
imgs_list_true = get_imgs(TRUENAME)

# define transformers
toTens = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tensor_list = trans_img(imgs_list)
tensor_list_true = trans_img(imgs_list_true)

inception_mdl = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

if is_cuda:
    inception_mdl = inception_mdl.to(device)

inception_mdl.eval()
# extract train and eval layers from the model
train_nodes, eval_nodes = get_graph_node_names(inception_mdl)
# remove the last layer
return_nodes = eval_nodes[:-1]

# create a feature extractor for each intermediary layer
feat_inception = create_feature_extractor(inception_mdl, return_nodes=return_nodes)
if is_cuda:
    feat_inception = feat_inception.to(device)

## metric
err_distr = get_distr(tensor_list, feat_inception)
true_distr = get_distr(tensor_list_true, feat_inception)

fid = calculate_fid(err_distr, true_distr)
with open("./generated/fids1.csv", "a") as f:
    f.write(ERRNAME + ", " + TRUENAME + ", " + "%.3f\n" % fid)
