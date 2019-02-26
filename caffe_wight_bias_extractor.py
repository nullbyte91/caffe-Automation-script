# -*- coding: utf-8 -*-

import caffe
import numpy as np
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototxt", help="model prototxt path .prototxt")
    parser.add_argument("--model", help="caffe model weights path .caffemodel")
    parser.add_argument("--output", help="output path")
    args = parser.parse_args()
    return args, parser

global args, parser
args, parser = parse_args()

def caffe2np(prototxt, model, output):
    Net = net = caffe.Net(prototxt, model, caffe.TEST)
    npNet = []
    for li in range(len(Net.layers)):  # for each layer in the net
        layer = {}  # store layer's information
        layer['name'] = Net._layer_names[li]
        # for each input to the layer (aka "bottom") store its name and shape
        layer['bottoms'] = [(Net._blob_names[bi], Net.blobs[Net._blob_names[bi]].data.shape)
                             for bi in list(Net._bottom_ids(li))]
        # for each output of the layer (aka "top") store its name and shape
        layer['tops'] = [(Net._blob_names[bi], Net.blobs[Net._blob_names[bi]].data.shape)
                          for bi in list(Net._top_ids(li))]
        layer['type'] = Net.layers[li].type  # type of the layer
        # the internal parameters of the layer. not all layers has weights.
        layer['weights'] = [Net.layers[li].blobs[bi].data[...]
                            for bi in range(len(Net.layers[li].blobs))]
        npNet.append(layer)
        
    caffe_parm = []
    for i, layer in enumerate(npNet):
        if layer['type'] == 'Convolution':
            weight = layer['weights'][0]
            bias = layer['weights'][1]
            caffe_parm.append({'w':weight, 'b':bias})
    
    print(type(caffe_parm))

    with open('parrot.pkl', 'wb') as f:
        pickle.dump(caffe_parm, f)
        f.close()

def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n caffe2np -h")

def main():
    print(args)
    if args.prototxt == None or args.model == None or args.output == None:
        usage_info()
        return None
    caffe2np(args.prototxt, args.model, args.output)

if __name__ == '__main__':
    main()