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
import pdb
import sys
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import plots
from basenji import trainer
from basenji import dataset
from basenji import rnann

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
saluki_test.py

Test the accuracy of a trained model.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head to test [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--save', dest='save',
      default=False, action='store_true',
      help='Save targets and predictions numpy arrays [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-g', dest='genes_file',
      default=None, type='str',
      help='File specifying gene names and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # inputs

  # read targets
  if options.genes_file is None:
    options.genes_file = '%s/genes.csv' % data_dir
  genes_df = pd.read_csv(options.genes_file, index_col=0)

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  # construct eval data
  eval_data = dataset.ExonDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode='eval')

  # initialize model
  seqnn_model = rnann.RnaNN(params_model)
  seqnn_model.restore(model_file, options.head_i)
  seqnn_model.build_ensemble(options.shifts)

  #######################################################
  # evaluation

  # evaluate
  test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data)

  # print summary statistics
  # This is the average over all the targets 
  print('\nTest Loss:         %7.5f' % test_loss)
  print('Test PearsonR:     %7.5f' % test_metric1.mean())
  print('Test R2:           %7.5f' % test_metric2.mean())

  # write target-level statistics
  # so metric for every cell type
  targets_acc_df = pd.DataFrame(np.transpose(np.vstack((np.squeeze(genes_df.columns[2:]),
                                                       np.squeeze(test_metric1),
                                                       np.squeeze(test_metric2)))),
                                columns=['target', 'pearsonR', 'R2']
    )

  targets_acc_df.to_csv('%s/acc.txt'%options.out_dir, sep='\t',
                        index=False, float_format='%.5f')

  #######################################################
  # predict

  if options.save:
    # compute predictions
    test_preds = seqnn_model.predict(eval_data).astype('float64')

    # read targets
    test_targets = eval_data.numpy(return_inputs=False)

    # read genes
    gene_ids = np.array(genes_df.index[genes_df['split'] == options.split_label], dtype='S')

    preds_h5 = h5py.File('%s/preds.h5' % options.out_dir, 'w')
    preds_h5.create_dataset('preds', data=test_preds)
    preds_h5.create_dataset('genes', data=gene_ids)
    preds_h5.close()

    targets_h5 = h5py.File('%s/targets.h5' % options.out_dir, 'w')
    targets_h5.create_dataset('targets', data=test_targets)
    targets_h5.create_dataset('genes', data=gene_ids)

    targets_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
