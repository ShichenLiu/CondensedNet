#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 给定layer_name, 任选channel, 任选100张图片，计算这100张图片相应最大神经元的位置，作为给定neuron位置
# 计算imagenet validation所有图片在给定neuron的响应，选取最大9张，将最大9张的图片序号保存在相应txt文件中
# 先用generate_run_code，生成所有该次演示的代码


# tfapp
# python generate_plot_new.py --txt_name=dense_block2__conv_block1__x2__c_8

# 问题：
# 0. 找到图片对应的感受野并剪裁放大
# 1. 画neuron的导数比较合适还是画mask的导数比较合适
# 2. image的index是否需要加一

# Tensorflow Graph & Graph Def
# protobuf: transfer Graph to C++
# GraphDef: serialized version of Graph
# read *pb using GraphDef and bind the GraphDef to a default Graph
# use a session to run the Graph for computation

# 20191111
# method1: image-label being give
# method2: highest pre-softmax

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import re
import ast
import scipy.misc
import numpy as np
import tensorflow as tf
from densenet_preprocessing import *
# from utility import *
from plot import *
import argparse
from matplotlib import pyplot as plt
from densenet import *
from PIL import Image
from ImageNet_val_class_mapping.imagenet_validation_predict import get_imagenet_val_id

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def combine_two_images(image, value, image_save_dir, image_num):
    image_path = os.path.join(image_save_dir, 'image.jpg')
    backprop_path = os.path.join(image_save_dir, 'backprop.jpg')
    scipy.misc.imsave(image_path, image)
    scipy.misc.imsave(backprop_path, value)

    height = image.shape[0] # 79
    width = image.shape[1]  # 53
    # 注意：这里是 image.shape: (79,53)
    # 但是 image.size: (53,79) # 通过 Image.open(path) 转化以后
    
    result = Image.new("RGB", (width * 2, height))
    combine_path = [image_path, backprop_path]

    for index, file in enumerate(combine_path):
      path = os.path.expanduser(file)
      img = Image.open(path)
      x = index // 1 * width
      y = index // 2 * height
      # w, h = img.size
      print('pos {0},{1} size {2},{3}'.format(x, y, width, height))
      result.paste(img, (x, y, x + width, y + height))

    res_path = os.path.join(image_save_dir, str(image_num)+'.jpg')

    # remove image if it already exists
    if os.path.exists(res_path):
        os.remove(res_path)
    print('Result image save to path {}'.format(res_path))
    result.save(os.path.expanduser(res_path))

    # delete tmp image and backprop file
    os.remove(image_path)
    os.remove(backprop_path)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--txt_name', type = str, help='name of visualized layer and channel')
   args = parser.parse_args()
   graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'
  
   # find class index 
   path_uid_to_labels = './ImageNet_val_class_mapping/imagenet_2012_validation_synset_labels.txt' # validation_uid
   path_uid_to_name = './ImageNet_val_class_mapping/imagenet_synset_to_human_label_map.txt'
   path_cls_to_name_panda_389 = './ImageNet_val_class_mapping/imagenet1000_clsidx_to_labels.txt'
   
   image_dir_val = r'E:\denseNet\densenet_visualize_crop\ImageNet_val_class_mapping\imagenet_val_first10'
   txt_name = 'logits'
   block_dir = 'logits'
   all_images = os.listdir(image_dir_val)
   filenames = [os.path.join(image_dir_val, path) for path in all_images]
   image_save_dir = os.path.join(block_dir, 'images_tmp')

   img_num = 2
   image_path = filenames[img_num]
   print('*** image_path={}'.format(image_path))
   image_preprocessed = get_image_preprocessed(image_path) # (1,224,224,3)
   print('*** image_preprocessed.shape', image_preprocessed.shape)
   # 按照验证集的给定标签进行可视化
   # c_ind = get_imagenet_val_id(image_path,path_uid_to_labels,path_uid_to_name,path_cls_to_name_panda_389)
   # model predicted output
   # pre_label = tf.nn.top_k(tf.nn.softmax(out_act), k=5, sorted=True)
   
   @tf.RegisterGradient("GuidedRelu")
   def _GuidedReluGrad(op, grad):
      gate_g = tf.cast(grad > 0, "float32")
      gate_y = tf.cast(op.outputs[0] > 0, "float32")
      return grad * gate_g * gate_y
   
   input_name = "Placeholder:0"
   # layer_name = "densenet121/predictions/Softmax:0" # visualize biggest activation

   # dense_block2_conv_block1_channel
   layer_name = "densenet121/dense_block2/conv_block1/x2/Conv/convolution:0"
   c_ind = 1
   # load graph
   def get_default_graph(graph_pb_path):
       config = tf.ConfigProto(allow_soft_placement = True)
       with tf.gfile.GFile(graph_pb_path, "rb") as f:
              graph_def = tf.GraphDef()
              graph_def.ParseFromString(f.read())
       with tf.Graph().as_default() as graph:
              tf.import_graph_def(graph_def, name="")
              print('*** Reload graph done')
       return graph                 
   
   # load graph weights from ckpt file
   # graph = get_default_graph(graph_pb_path)
   graph = tf.get_default_graph()
   with graph.gradient_override_map({'Relu':'GuidedRelu'}):
          # graph = get_default_graph(graph_pb_path)
          # config = tf.ConfigProto(allow_soft_placement = True)
          # load pre-trained model
          config = tf.ConfigProto(allow_soft_placement = True)
          with tf.gfile.GFile(graph_pb_path, "rb") as f:
              graph_def = tf.GraphDef()
              graph_def.ParseFromString(f.read())
          # 问题出在这里，这里重新定义了一个graph
          # 这里的可视化与VGG不同是因为这里是用DenseNet预训练的模型
          # 这里应该用override 的 graph, 而不是 重新定义的tf.Graph
          # with tf.Graph().as_default():
          with graph.as_default():
              tf.import_graph_def(graph_def, name="")
              print('*** Reload graph done')
          inputs = graph.get_tensor_by_name(input_name)
          layer = graph.get_tensor_by_name(layer_name)
          
          # mask out all other values
          # mask = np.zeros((1,height,width,1))
          # mask[0,h_ind, w_ind, 0] = 1
          # layer_new = tf.multiply(layer, mask)

          layer_new = tf.reduce_max(layer)
          print('*** layer_new={}'.format(layer_new))
          
          guided_back_pro = tf.gradients(layer_new, inputs)[0]
          print('*** guided_back_pro', guided_back_pro)
   
   with tf.Session(graph=graph) as sess:
        value = sess.run(guided_back_pro, {inputs:image_preprocessed})
   
   # visualization guided_backprop
   value_cropped = np.squeeze(value[0])
   image_cropped = np.squeeze(image_preprocessed)
   # ===============================  crop and plot
   final_save_dir = os.path.join(r'E:\denseNet\densenet_visualize_crop\layer_new\logits', txt_name)
   if not os.path.exists(final_save_dir):
      os.makedirs(final_save_dir)
   combine_two_images(image_cropped, value_cropped, final_save_dir, str(img_num))


