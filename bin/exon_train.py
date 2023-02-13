#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
from optparse import OptionParser

import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from basenji import dataset
from basenji import rnann
from basenji import trainer

"""
saluki_train.py

Train Saluki model using given parameters and data on RNA sequence.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)
  parser.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rbp', dest='rbp_track',
      default=0, type=int,
      help='Whether to use RBP track as input')
  parser.add_option('--seq', dest='seq_track',
      default=1, type=int,
      help='Whether to use seq track as input')
  parser.add_option('--splice', dest='splice_track',
      default=1, type=int,
      help='Whether to use splice track as input')
  (options, args) = parser.parse_args()

  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = args[0]
    data_dirs = args[1:]

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  os.makedirs(options.out_dir, exist_ok=True)
  if params_file != '%s/params.json' % options.out_dir:
    shutil.copy(params_file, '%s/params.json' % options.out_dir)

  # read datasets
  train_data = []
  eval_data = []
  
  if options.rbp_track:
      for data_dir in data_dirs:
        # load train data
        train_data.append(dataset.ExonRBPDataset(data_dir,
          split_label='train',
          batch_size=params_train['batch_size'],
          shuffle_buffer=params_train.get('shuffle_buffer', 1024),
          mode='train', splice_track=options.splice_track,
          seq_track=options.seq_track))
    
        # load eval data
        eval_data.append(dataset.ExonRBPDataset(data_dir,
          split_label='valid',
          batch_size=params_train['batch_size'],
          mode='eval', splice_track=options.splice_track,
          seq_track=options.seq_track))
  else:
      for data_dir in data_dirs:
        # load train data
        train_data.append(dataset.ExonDataset(data_dir,
          split_label='train',
          batch_size=params_train['batch_size'],
          shuffle_buffer=params_train.get('shuffle_buffer', 1024),
          mode='train',
          splice_track=options.splice_track))
    
        # load eval data
        eval_data.append(dataset.ExonDataset(data_dir,
          split_label='valid',
          batch_size=params_train['batch_size'],
          mode='eval',
          splice_track=options.splice_track))
        

  if options.seq_track == 0:
      params_model['seq_depth'] -= 4
  if options.splice_track == 0:
      params_model['seq_depth'] -= 1

    # initialize model
  seqnn_model = rnann.RnaNN(params_model)

  # initialize trainer
  seqnn_trainer = trainer.Trainer(params_train, train_data, 
                                  eval_data, options.out_dir)
  # compile model
  seqnn_trainer.compile(seqnn_model)

  # fit
  if len(data_dirs) == 1:
    seqnn_trainer.fit_tape(seqnn_model)
  else:
    seqnn_trainer.fit2(seqnn_model)
    
  ## When finished
  fn = options.out_dir.replace('/', '_')
  with open(fn, 'x') as f:
    f.write('Finished!')



################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  
