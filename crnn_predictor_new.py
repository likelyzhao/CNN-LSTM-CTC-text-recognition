#!/usr/bin/env python2.7
# coding=utf-8
from __future__ import print_function
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("../../amalgamation/python/")
sys.path.append("../../python/")
import argparse

#`from mxnet_predict import Predictor
import mxnet as mx

from symbol.crnn import crnn

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


class Predictor(object):
    def __init__(self, path_of_json, path_of_params, classes, data_shape, batch_size, num_label, num_hidden, num_lstm_layer,ctx=0,
                 model_prefix="deploy",model_epoch=0):
        super(Predictor, self).__init__()
        self.path_of_json = path_of_json
        self.path_of_params = path_of_params
        self.classes = classes
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.num_hidden = num_hidden
        self.num_lstm_layer = num_lstm_layer
        self.predictor = None
        self.ctx = ctx
        self.model_prefix = model_prefix
        self.model_epoch = model_epoch
        self.__init_model()

    def __init_model(self):
#        init_c = [('l%d_init_c'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
        init_h = [('l%d_init_h'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
     #   init_states = init_c + init_h
        init_states =  init_h

        all_shapes = [('data', (batch_size, 1, self.data_shape[1], self.data_shape[0]))] + init_states + [('label', (self.batch_size, self.num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]

        sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_prefix, self.model_epoch)
        print(sym.get_children().get_children().get_children())
        if "pred" in  sym.list_arguments():
            sym_out = sym.list_arguments()['pred']
        else:
            sym_out = sym
            sym_out = sym.get_children().get_children().get_children()

        model = mx.mod.Module(symbol=sym_out, context=mx.gpu(0), data_names=['data','l0_init_h','l1_init_h','l2_init_h','l3_init_h'],label_names=None)
        print(model)
        model.bind(force_rebind=True ,for_training=False, data_shapes=[
                      ('data', (batch_size, 1, self.data_shape[1], self.data_shape[0])),
		      ('l0_init_h',(self.batch_size, self.num_hidden)),
                      ('l1_init_h',(self.batch_size, self.num_hidden)),
                      ('l2_init_h',(self.batch_size, self.num_hidden)),
                      ('l3_init_h',(self.batch_size, self.num_hidden)),
                      ],label_shapes=None)
        
 #       model.bind(for_training=False, data_shapes=[
#                      ('data', (batch_size, 3, self.data_shape[1], self.data_shape[0])),
#		      ('l1_init_h',(self.batch_size, self.num_hidden)),
#                      ('l2_init_h',(self.batch_size, self.num_hidden)),
#                      ('l3_init_h',(self.batch_size, self.num_hidden)),
#                      ('l4_init_h',(self.batch_size, self.num_hidden))],label_shapes=[('label',(self.batch_size, self.num_label))])

        # load model parameters
        print(model.output_shapes)
        model.set_params(arg_params, aux_params, allow_missing=True)
        self.model = model

#        self.output()




#       self.predictor = Predictor(open(self.path_of_json).read(),
#                                    open(self.path_of_params).read(),
#                                    all_shapes_dict,dev_type="gpu", dev_id=0)


class lstm_ocr_model(object):
    # Keep Zero index for blank. (CTC request it)
    def __init__(self, path_of_json, path_of_params, classes, data_shape, batch_size, num_label, num_hidden, num_lstm_layer):
        super(lstm_ocr_model, self).__init__()
        self.path_of_json = path_of_json
        self.path_of_params = path_of_params
        self.classes = classes
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.num_hidden = num_hidden
        self.num_lstm_layer = num_lstm_layer
        self.predictor = None
        self.__init_ocr()

    def __init_ocr(self):
        init_c = [('l%d_init_c'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
        init_h = [('l%d_init_h'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
        init_states = init_c + init_h

        all_shapes = [('data', (batch_size, 1, self.data_shape[1], self.data_shape[0]))] + init_states + [('label', (self.batch_size, self.num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json).read(),
                                    open(self.path_of_params).read(),
                                    all_shapes_dict,dev_type="gpu", dev_id=0)

def forward_ocr(ocr_model, imgbgr,datashape,num_hidden,classes):
    img = cv2.cvtColor(imgbgr,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, datashape)
    img = img.reshape((1,1, datashape[1], datashape[0]))
    img = np.multiply(img, 1/255.0)
#    from collections import namedtuple
#    Batch = namedtuple('Batch', ['data','label'],['label'])
#    ocr_model.model.forward(Batch(img,mx.nd.zeros((1,9))))
#    test_intr = Batch((img),(mx.nd.zeros((1,9))))
    test_intr = mx.io.DataBatch(data=[mx.nd.array(img),mx.nd.zeros((1,num_hidden)),mx.nd.zeros((1,num_hidden)),mx.nd.zeros((1,num_hidden)),mx.nd.zeros((1,num_hidden))],label=None)
    ocr_model.model.forward(test_intr)
    prob = ocr_model.model.get_outputs()[0].asnumpy()
#    for prob in probs:
#    print(prob)
    
    label_list = []
    for p in prob:
 #       print(p)
        max_index = np.argsort(p)[::-1][0]
        label_list.append(max_index)
    return _get_string(label_list,classes)

def _get_string( label_list,classes):
    # Do CTC label rule
    # CTC cannot emit a repeated symbol on consecutive timesteps
    ret = []
    label_list2 = [0] + list(label_list)
    for i in range(len(label_list)):
        c1 = label_list2[i]
        c2 = label_list2[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    # change to ascii
    s = ''
    for l in ret:
        if l > 0 and l < (len(classes)+1):
            c = classes[l-1]
        else:
            c = ''
        s += c
    return s

def parse_args():
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--img', dest='img', help='which image to use',
                        default=os.path.join(os.getcwd(), 'data', 'test', '1.jpg'), type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    json_path = os.path.join(os.getcwd(), 'model', 'crnn_ctc-symbol.json')
    param_path = os.path.join(os.getcwd(), 'model', 'crnn_ctc-0100.params')
#    mx.nd.load('crnn_ctc-0099.params')
    num_label = 9 # Set your max length of label, add one more for blank
    batch_size = 1
    num_hidden = 256
    num_lstm_layer = 2
    data_shape = (100, 32)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", 
        "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    demo_img = args.img

#    _lstm_ocr_model = lstm_ocr_model(json_path, param_path, classes, data_shape, batch_size,
#                                    num_label, num_hidden, num_lstm_layer)
    _lstm_ocr_model = Predictor(json_path, param_path, classes, data_shape, batch_size,
                                   num_label, num_hidden, num_lstm_layer,model_prefix="model/crnn_ctc",model_epoch=100)
    print(demo_img)
    img = cv2.imread(demo_img)
    #img = cv2.bitwise_not(img)
#    print(img)
    _str = forward_ocr(_lstm_ocr_model,img,data_shape,num_hidden,classes)
    print('Result: ', _str)
#    plt.imshow(img)
#    plt.gca().text(0, 6.8,
#                    '{:s} {:s}'.format("prediction", _str),
                    #bbox=dict(facecolor=colors[cls_id], alpha=0.5),
#                    fontsize=12, color='red')
 #   plt.show()
