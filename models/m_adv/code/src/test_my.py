#!/usr/bin/env python
import sys
import os
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import mxnet as mx
import numpy as np
import cv2
from vgg_mx.symbol_vgg import VGG
from collections import namedtuple
import symbol_sentiment
from model import *
import os

def generate_poem(img_feature):
    return model.test_one_image(img_feature)

def data_trans(img, shape, mu):
    transformed_img = img.transpose((2, 0, 1)) * 255
    transformed_img -= mu[:, np.newaxis, np.newaxis];
    return transformed_img

def crop_lit_centor(img, mu, img_len = 224):
    [n,m,_]=img.shape
    if m>n:
        m = int(m*256/n)
        n = 256
    else:
        n = int(n*256/m)
        m = 256
    return data_trans(cv2.resize(img,(m,n))/255.0, shape=(1,3,n,m), mu=mu)[:,int((n-img_len)/2):int((n+img_len)/2),int((m-img_len)/2):int((m+img_len)/2)]

def get_mod(output_name = 'relu7_output', sym = None, img_len = 224):
    if sym is None:
        vgg = VGG()
        sym = vgg.get_symbol(num_classes = 1000, 
                  blocks = [(2, 64),
                            (2, 128),
                            (3, 256), 
                            (3, 512),
                            (3, 512)])
        internals = sym.get_internals()
        sym = internals[output_name]

    mod = mx.module.Module(
            context = ctx,
            symbol = sym,
            data_names = ("data", ),
            label_names = ()
    )

    mod.bind(data_shapes = [("data", (1, 3, img_len, img_len))], for_training = False)

    return mod


def get_obj_feature(img):
    mu = np.array([104,117,123])
    transformed_img = crop_lit_centor(img, mu)
    transformed_img = transformed_img[None]
    object_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
    outputs = object_model.get_outputs()[0].asnumpy()
    return outputs

def get_scene_feature(img):
    mu = np.array([105.487823486,113.741088867,116.060394287])
    transformed_img = crop_lit_centor(img, mu)
    transformed_img = transformed_img[None]
    scene_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
    outputs = scene_model.get_outputs()[0].asnumpy()
    return outputs

def get_sentiment_feature(img):
    mu = np.array([97.0411,105.423,111.677])
    transformed_img = crop_lit_centor(img, mu, img_len = 227)
    transformed_img = transformed_img[None]
    sentiment_model.forward(Batch([mx.nd.array(transformed_img)]), is_train = False)
    outputs = sentiment_model.get_outputs()[0].asnumpy()
    return outputs

def extract_feature(image_file):
    img = cv2.imread(image_file)
    assert img is not None, IOError(
            'The file `{}` may be not an image'.format(image_file))
    # img.shape: H, W, T
    if img.ndim == 2:
        # gray image
        img = np.stack([img, img, img], axis=2)
    else:
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            if img.shape[2] == 4:
                # remove alpha channel
                img = img[:, :, :3]
        else:
            raise Exception('Invalid Image `{}` whose shape is {}'.format(image_file, img.shape))
    obj_feat = get_obj_feature(img)
    scene_feat = get_scene_feature(img)
    sentiment_feat = get_sentiment_feature(img)
    image_features = [obj_feat, sentiment_feat, scene_feat]
    img_feature = np.hstack(image_features)
    return img_feature

if __name__ == '__main__':
    ctx = [mx.cpu()]

    feature_names = ['object', 'scene', 'Sentiment']
    Batch = namedtuple('Batch', ['data'])

    object_model = get_mod()
    object_model.load_params('../model/object.params')

    scene_model = get_mod()
    scene_model.load_params('../model/scene.params')

    sentiment_model = get_mod(sym = symbol_sentiment.get_sym(), img_len = 227)
    sentiment_model.load_params('../model/Sentiment.params')

    ckpt_file = '../model/ckpt/epoch03_lr0.000005.ckpt'
    sess_config = tf.ConfigProto()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpus)
    sess = tf.InteractiveSession(config=sess_config)
    batch_size = 1
    model = SeqGAN(sess, batch_size)

    model.load_params(ckpt_file)

    with open("test_poem2img.json", "r") as file:
        test_data = json.load(file)["poem2img"]
    
    results = {
        "Mean CLIP Score" : 0,
        "Median CLIP Score" : 0,
        "Max CLIP Score" : 0,
        "Min CLIP Score" : 0,
        "Mean Distinct-2 Score" : 0,
        "Median Distinct-2 Score" : 0,
        "Max Distinct-2 Score" : 0,
        "Min Distinct-2 Score" : 0,
        "Mean Imageability Score" : 0,
        "Median Imageability Score" : 0,
        "Max Imageability Score" : 0,
        "Min Imageability Score" : 0,
        "Poems" : [],
        "CLIP Scores" : [],
        "Distinct-2 Scores" : [],
        "Imageability Scores" : []
    }
    
    for i in tqdm(range(len(test_data))):
        path = "image_test/" + str(i+1) + ".png"
        img_feature = extract_feature(path)
        results["Poems"].append('\n' + generate_poem(img_feature)[0].replace('\n', '\n') + '\n')
    
    with open("poems.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    print("Hello world")