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
import gc
from pathlib import Path


import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')

from basenji import dataset
from basenji import rnann
from basenji import stream


if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
exon_ism_tfr.py

ISM for the all test sequences.
"""

def main():
  usage = 'usage: %prog [options] <model_dir> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('--head', dest='head_i',
      default=0, type='int',
      help='Parameters head to test [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='ism_out',
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

  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--multihead', dest='multihead',
      default=False,
      help='Whether a multihead model was used during training')
  parser.add_option('--exons', dest='exons',
                    default=None, 
                    help='Which exons to do ISM for')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide model, and test data HDF5')
  else:
    model_dir = args[0]
    data_dir = args[1]
    
    
  Path(options.out_dir).mkdir( parents=True, exist_ok=True )

  #######################################################
  # inputs

  # read targets
  genes_file = '%s/genes.csv' % data_dir
  genes_df = pd.read_csv(genes_file, index_col=0)
  gene_ids = np.array(genes_df.index[genes_df['split'] == options.split_label], 
                      dtype='S')
  gene_ids2 = np.array(genes_df.index[genes_df['split'] == options.split_label])
  
  # read exons
  if options.exons != None:
     exons = pd.read_csv(options.exons, index_col=0)

  # read model parameters
  params_file = '%s/params.json' % model_dir
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  if options.rbp_track:
    # load eval data
    eval_data =dataset.ExonRBPDataset(data_dir,
        split_label=options.split_label,
        batch_size=1,
        mode='eval', splice_track=options.splice_track,
        seq_track=options.seq_track)
    
  else:
    # load eval data
    eval_data = dataset.ExonDataset(data_dir,
        split_label=options.split_label,
        batch_size=1,
        mode='eval',
        splice_track=options.splice_track)
          
  # initialize model
  if options.multihead:
      model_file = model_dir + '/model' + str(options.head_i) + '_best.h5'
  else:
      model_file = '%s/model_best.h5' % model_dir
  print(model_file)
  print(options.head_i)
  
  if options.seq_track == 0:
      params_model['seq_depth'] -= 4
  if options.splice_track == 0:
      params_model['seq_depth'] -= 1
  
  seqnn_model = rnann.RnaNN(params_model)
  seqnn_model.restore(model_file, head_i=int(options.head_i))
  
  # make sequence generator
  print('Generating mutations started')
  seq_length = params_model['seq_length']
  seqs_gen = satmut_gen(eval_data, seq_length)
  print('Generating mutations finished')

  #################################################################
  # setup output
  
  num_seqs = eval_data.num_seqs

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('genes', data=gene_ids)
  scores_h5.create_dataset('seqs', dtype='bool',
      shape=(num_seqs, seq_length, 4))
  scores_h5.create_dataset('splice', dtype='bool',
      shape=(num_seqs, seq_length))
  scores_h5.create_dataset('ref', dtype='float16',
      shape=(num_seqs, 1))
  scores_h5.create_dataset('ism', dtype='float16',
      shape=(num_seqs, seq_length, 4))

  # store mutagenesis coordinates?


  #################################################################
  # predict scores, write output

  # initialize predictions stream
  batch_size = 2*params_train['batch_size']
  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen,
    batch_size, stream_seqs=8*batch_size)

  # sequence index
  si = 0

  # predictions index
  pi = 0

  for seq_1hotc, _ in eval_data.dataset:
      
    mutate = True 
    # Check if we need to do ISM for this gene
    if options.exons != None:
       current_gene = gene_ids2[si]
       if np.isin(current_gene, exons) == False:
           print(current_gene)
           mutate = False
      
    # convert to single numpy 1hot
    seq_1hotc = seq_1hotc.numpy().astype('bool')[0]

    print('Predicting %d, %d nt' % (si,seq_length), flush=True)
    print(pi)

    # write reference sequence  
    seq_1hotc_mut = seq_1hotc[0:seq_length]
    seq_1hot_mut = seq_1hotc_mut[:,:4]
    scores_h5['seqs'][si,:,:] = seq_1hot_mut
    scores_h5['splice'][si,:] = seq_1hotc_mut[:,4]

    # initialize scores
    seq_scores = np.zeros((seq_length, 4), dtype='float32')

    # collect reference prediction
    preds_mut0 = preds_stream[pi]
    pi += 1

    # for each mutated position
    for mi in range(seq_length):
      # if position has nucleotide
      if seq_1hot_mut[mi].max() < 1:
        # reference score
        seq_scores[mi,:] = preds_mut0
      else:
        # for each nucleotide
        for ni in range(4):
          if seq_1hot_mut[mi,ni]:
            # reference score
            seq_scores[mi,ni] = preds_mut0
          else:
            # collect and set mutation score
            if mutate:
                seq_scores[mi,ni] = preds_stream[pi]
            pi += 1

    # normalize
    seq_scores -= seq_scores.mean(axis=1, keepdims=True)

    # write to HDF5
    if mutate:
       scores_h5['ref'][si] = preds_mut0.astype('float16')
       scores_h5['ism'][si,:,:] = seq_scores.astype('float16')

    # increment sequence
    si += 1

    # clean memory
    gc.collect()
    
  # close output HDF5
  scores_h5.close()
  
  
def satmut_gen(eval_data, mut_len):
  """Construct generator for 1 hot encoded saturation
     mutagenesis DNA sequences."""

  # taa1 = dna_io.dna_1hot('TAA')
  # tag1 = dna_io.dna_1hot('TAG')
  # tga1 = dna_io.dna_1hot('TGA')

  for seq_1hotc, _ in eval_data.dataset:
    seq_1hotc = seq_1hotc.numpy()[0]
    yield seq_1hotc

    # for mutation positions
    for mi in range(0, mut_len):
      # if position as nucleotide
      if seq_1hotc[mi].max() == 1:
        # for each nucleotide
        for ni in range(4):
          # if non-reference
          if seq_1hotc[mi,ni] == 0:
            # copy and modify
            seq_mut_1hotc = np.copy(seq_1hotc)
            seq_mut_1hotc[mi,:4] = 0
            seq_mut_1hotc[mi,ni] = 1

            yield seq_mut_1hotc


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  
  
  
  
  
  
