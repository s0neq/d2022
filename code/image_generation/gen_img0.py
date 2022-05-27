# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

from rudalle.pipelines import generate_images, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_ruclip
from rudalle.utils import seed_everything


def _get_logger():
    """
    Initialize logger
    """
    logger = logging.getLogger('gen_img')
    logger.setLevel(logging.DEBUG)
    log_handler = RotatingFileHandler(
        "./gen_img.log", 
        maxBytes=100*1024*1024,
        backupCount=5
    )
    log_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]:\t%(message)s'))
    logger.addHandler(log_handler)

    return logger

logger = _get_logger()

os.makedirs("./generated/", exist_ok=True)

device = 'cuda:0'
tokenizer = get_tokenizer()
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)

vae = get_vae().to(device)
ruclip, ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
ruclip = ruclip.to(device)

img_ids_seen = dict()


def one_domain3(df_domain, name):
    """
    saves jpgs to ./generated/+name
    filename format: captionID_imgID_repetitionNUM.jpg
    """
    caps = list(df_domain.sent)
    img_ids = list(df_domain.img_id)
    caption_ids = list(df_domain.caption_id)

    for text, img_id, caption_id in zip(caps, img_ids, caption_ids):
        seed_everything(42)
        top_k, top_p, images_num = 128, 0.95, 3
        _pil_images, _scores = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p)
        top_images, _ = cherry_pick_by_clip(_pil_images, text, ruclip, ruclip_processor, device=device, count=1)
        
        if (caption_id, img_id) not in img_ids_seen:
          img_ids_seen[(caption_id, img_id)] = 0
        img_ids_seen[(caption_id, img_id)] += 1

        tofile = str(img_id)
        if img_ids_seen[(caption_id, img_id)] != 1:
          tofile += "_" + str(img_ids_seen[(caption_id, img_id)])

        filename = str(caption_id) + "_" + tofile + ".jpg"
        os.makedirs("./generated/"+name+"/", exist_ok=True)
        path = "./generated/"+name+"/"
        top_images[0].save(path + filename , "JPEG")

try:
    logger.info("reading data.")
    name = "nsubj_Gender_error"
    df = pd.read_csv("https://raw.githubusercontent.com/s0neq/d2022/main/"+name+".csv", index_col=False)
    logger.info("data read.")
except:
    logger.error('couldnt get data')

try:
    logger.info("Starting generation.")
    one_domain3(df, name)
    logger.info("Success.")
except:
    logger.error('smth went wrong')
