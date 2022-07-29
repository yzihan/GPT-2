#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import threading
import model, sample, encoder


tf.compat.v1.disable_eager_execution()
models_dir = '/gpt-2/models'
models_dir = os.path.expanduser(os.path.expandvars(models_dir))
model_name='117M'  #model_name=124M : String, which model to use
seed=None #seed=None : Integer seed for random number generators, fix seed to reproduce results
nsamples=1 #nsamples=1 : Number of samples to return total
batch_size = 1 # batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
length=20 #length=None : Number of tokens in generated text, if None (default), is determined by model hyperparameters
temperature = 1 # temperature=1 : Float value controlling randomness in boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions.
top_k = 40 # top_k=40 : Integer value controlling diversity. 1 means only 1 word is considered for each step (token), resulting in deterministic completions, while 40 means 40 words are considered at each step. 0 (default) is a special setting meaning no restrictions. 40 generally is a good value.
top_p=1
assert nsamples % batch_size == 0

enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))
if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
 
import pprint 
import sys  
import copy
from time import time
from typing import Type
# sys.path.append("../")
import os
import json
from flask import Flask, jsonify, request, Blueprint, render_template, abort, send_from_directory
import re

# Initialize the app
app = Flask(__name__)

def requestParse(req_data):
    """解析请求数据并以json形式返回"""
    if req_data.method == "POST":
        if req_data.json != None:
            data = req_data.json
        else:
            data = req_data.form
    elif req_data.method == "GET":
        data = req_data.args
    return data
prompList = None
maxLength = None
doneFlag = True
@app.route('/continuePrompt', methods=['POST'])
# @cross_origin()
def continuePrompt():
    global prompList
    global maxLength
    global resultList
    global doneFlag
    prompList = requestParse(request)['prompList']
    maxLength = requestParse(request)['maxLength']
    resultList = []
    doneFlag = False
    i = 0
    while doneFlag == False:
        if i%1000==0:
            print("waiting for compute...")
        i+=1
    return jsonify({"resultList":resultList})


    

###################### for compute thread  ##############################################################
class ComputeThread(threading.Thread):
    def __init__(self):
        super(ComputeThread,self).__init__()   

    def run(self):
        global threadStopFlag 
        global stateExistLock
        global prompList
        global maxLength
        global resultList
        global doneFlag
        threadStopFlag = False
        sess = tf.Session(graph=tf.Graph())
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
    
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)
            print("save!!!")
            while not threadStopFlag:
                print("enter while")
                if doneFlag == True:
                    continue
                print("should not get here!!!")
                for raw_text in prompList:
                    context_tokens = enc.encode(raw_text)
                    generated = 0
                    for _ in range(nsamples // batch_size):
                        out = sess.run(output, feed_dict={
                            context: [context_tokens for _ in range(batch_size)]
                        })[:, len(context_tokens):]
                        for i in range(batch_size):
                            generated += 1
                            text = enc.decode(out[i])
                            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                            print(text)
                            resultList.append(re.split('\.|\,|\;|\n|\u2026',text)[0] if re.split('\.|\,|\;|\n|\u2026',text)[0]!="" else text )
                            print("=" * 80) 
                doneFlag = True



            
                

                
if __name__ == "__main__":
    ComputeThread = ComputeThread()
    ComputeThread.start()
    app.run(host='0.0.0.0', debug=True, threaded=True, port=6006)
