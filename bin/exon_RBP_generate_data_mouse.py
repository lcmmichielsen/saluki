# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:03:57 2022

@author: lcmmichielsen
"""

import os
from pathlib import Path
import argparse
import json
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import pysam

import tensorflow as tf

os.chdir('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/saluki')

from basenji.dna_io import dna_1hot

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dir',            dest='dir',             default='/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/Data/HippData',
                    help='Directory to with PSI values')
parser.add_argument('--file_glia',        dest='file_glia',     default='PSI_glia_norm.csv',
                    help='File with glia PSI values')
parser.add_argument('--file_neur',        dest='file_neur',     default='PSI_neur_norm.csv',
                    help='File with neur PSI values')
parser.add_argument('--file_folds',       dest='file_folds',    default='exon_info_folds.csv',
                    help='File with predefined folds for the 10fold CV')
parser.add_argument('--species',        dest='species',     default='human',
                    help='human or mouse')
parser.add_argument('--out',            dest='out',             default='tfrecords_all',
                    help='Output directory for the TFRecords')
parser.add_argument('--var_threshold',     dest='var_threshold',          type=float,     default=0.0,
                    help='Filter exons based on minimum delta PSI between neurons and glia')
parser.add_argument('--max_length',     dest='max_length',          type=int,     default=3072,
                    help='Maximum input length to the model')
parser.add_argument('--num_layers',     dest='num_layers',          type=int,     default=4,
                    help='Number of convolution layers in the Saluki model (related to input length)')
parser.add_argument('--padding',     dest='padding',          type=int,     default=400,
                    help='Padding around the exon sequence')
parser.add_argument('--RBPs',           dest='RBPs',                type=str,       default="UCHL5,AQR,BUD13,HNRNPC,YBX3,PPIG",
                    help='Which RBPs to add as input')
parser.add_argument('--encode',         dest='encode',              type=str,       default='sparse',
                    help='How to encode the begin and end of the exon (either "complete" or "sparse"')
parser.add_argument('--cell_type',      dest='cell_type',           type=str,       default='both',
                    help='Which celltype') # 'glia' / 'neurons' / 'both'
parser.add_argument('--exon_list',      dest='exon_list',           type=str,       default='None',
                    help='Exons to include in tfrecords')
# If both, we generate three dir: 1. glia, 2. neuron, 3. multihead
# If glia or neurons, we generate only the glia or neuron dir.

args = parser.parse_args()

general_dir = args.dir
file_glia = args.file_glia
file_neur = args.file_neur
file_folds = args.file_folds
species = args.species
out_dir = args.out
cell_type = args.cell_type
var_threshold = args.var_threshold
max_length = args.max_length
num_layers = args.num_layers
padding = args.padding
encode = args.encode
exon_list = args.exon_list
RBPs = args.RBPs
if np.all(RBPs == '') == False:
    RBPs = np.asarray(args.RBPs.split(','))
    
if exon_list != 'None':
    exons_tokeep = pd.read_csv(exon_list, index_col=0)
    
general_out_dir = general_dir + '/' + out_dir

# Read both the neuronal and glia PSI values
# We need both for var threshold, even if we're interested in single-task model
PSI_glia = pd.read_csv(general_dir + '/' + file_glia, index_col = 0)
PSI_neur = pd.read_csv(general_dir + '/' + file_neur, index_col = 0)
exon_folds = pd.read_csv(general_dir + '/' + file_folds, index_col=0)

idx_tokeep = (PSI_neur['0'] >= 0) & (PSI_glia['0'] >= 0)
PSI_neur = PSI_neur.loc[idx_tokeep]
PSI_glia = PSI_glia.loc[idx_tokeep]
exon_folds.loc[idx_tokeep]

# Characteristics of the exons (needed to extract seq. later on)
exon_info = pd.DataFrame(PSI_glia.index)[0].str.split('_', expand=True)
exon_info = exon_info.rename(columns={0: 'chr', 1: 'start', 2: 'end',
                                      3: 'ENSG', 4: 'strand'})

## Load the peaks
if np.all(RBPs == '') == False:
    peaks_path = general_dir + '/RBP_coor_repl.pickle'
    with open(peaks_path, 'rb') as handle:
        RBP_coor_repl, rows, columns = pickle.load(handle)
    peaks = pd.DataFrame(RBP_coor_repl, index=rows, columns=columns)
else:
    peaks=0

# Function to generate the data
def create_tfrecords(PSI, exon_info, train_idxs, val_idxs, 
                     test_idxs, fold, general_out_dir, peaks,
                     max_length, num_layers, padding, encode):
    
    fold_dir = general_out_dir + '/fold' + str(num_fold)
    tfr_dir = fold_dir + '/tfrecords' 
    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    Path(tfr_dir).mkdir(parents=True, exist_ok=True)

    fold_indexes = [train_idxs, val_idxs, test_idxs]
    
    # Write the file genes.csv
    split = np.zeros((len(exon_info),1), dtype='<U5')
    split[train_idxs] = 'train'
    split[val_idxs] = 'valid'
    split[test_idxs] = 'test'
    target = PSI.values
    genes = pd.DataFrame(np.hstack((split,target)), 
                         index=PSI.index, columns=np.hstack((['split'], PSI.columns)))
    genes = genes[genes['split'] != '']
    genes.to_csv(fold_dir + '/genes.csv')
    
    # Options TFR writer
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    seqs_per_tfr = 256
    fold_labels = ['train', 'valid', 'test']
    max_seq_length = max_length
    padding = padding
    
    # open FASTA
    if species == 'human':
        fasta_file = '../Ref/GRCh38.primary_assembly.genome.fa'
    elif species == 'mouse':
        fasta_file = '../Ref/mm10.fa'
    fasta_open = pysam.Fastafile(fasta_file)
    
    def rc(seq):
        return seq.translate(str.maketrans("ATCGatcg","TAGCtagc"))[::-1]
    
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    # Write the TFRecords
    num_folds = 3 #train-val-test
    
    for fi in range(num_folds):
        exon_ID_set = PSI.index[fold_indexes[fi]]
        exon_info_set = exon_info.iloc[fold_indexes[fi]]
        PSI_val_set = PSI.iloc[fold_indexes[fi]]
        
        num_set = exon_ID_set.shape[0]
        
        ### So they split the sequences over different TFRs such that the file stays reasonable in size??
        num_set_tfrs = int(np.ceil(num_set / seqs_per_tfr)) 
        print(num_set_tfrs)
        
        # gene sequence index
        si = 0
    
        for tfr_i in range(num_set_tfrs):
            # Create the file e.g 'tfr_records/test-0.tfr'
            tfr_file = '%s/%s-%d.tfr' % (tfr_dir, fold_labels[fi], tfr_i)
            print(tfr_file)
            with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
                # TFR index
                ti = 0
                
                # This is to make sure that the max. genes per file stays below 256
                # And that for the last batch we stop on time (si < num_set)
                while ti < seqs_per_tfr and si < num_set:
                    # Get the genes that should be in this set
                    seq_chrm = exon_info_set['chr'].iloc[si]
                    seq_start = int(exon_info_set['start'].iloc[si])
                    seq_end = int(exon_info_set['end'].iloc[si])+1 #Because fasta.fetch doesn't include end bp
                    seq_strand = exon_info_set['strand'].iloc[si]
                    gene = exon_info_set['ENSG'].iloc[si]
                    
                    if (seq_end-seq_start) > max_seq_length:
                        seq_start_ = seq_start-100
                        seq_end_ = seq_start_ + max_seq_length
                        splicing_ind = np.array([100], dtype=np.int64)    
                    elif(seq_end-seq_start+2*padding) > max_seq_length:
                        pad_temp = np.floor((max_seq_length - (seq_end-seq_start+1))/2)
                        seq_start_ = seq_start - pad_temp
                        seq_end_ = seq_end + pad_temp
                        splicing_ind = np.array([pad_temp, pad_temp+seq_end+1-seq_start], dtype=np.int64)
                    else:
                        seq_start_ = seq_start - padding
                        seq_end_ = seq_end + padding
                        splicing_ind = np.array([padding, padding+seq_end+1-seq_start], dtype=np.int64)
                        
                    seq_dna = fasta_open.fetch(seq_chrm, seq_start_, seq_end_)
    
                    # verify length
                    len_DNA = len(seq_dna)
                    assert(len_DNA <= max_seq_length)
    
                    # orient
                    if seq_strand == '-':
                        seq_dna = rc(seq_dna)
    
                    # one hot code
                    seq_1hot = dna_1hot(seq_dna)
                    seq_len = np.array(len(seq_dna), dtype=np.int64)
                    
                    # splicing
                    splicing = np.zeros((seq_len,1), dtype=np.int8)
                    if encode == 'sparse':
                        splicing[splicing_ind] = 1
                    else:
                        if len(splicing_ind) == 2:
                            splicing[splicing_ind[0]:splicing_ind[1]] =1
                        else:
                            splicing[splicing_ind[0]:] = 1
                                
                    # get targets
                    targets = PSI_val_set.iloc[si].values
                    targets = targets.reshape((1,-1)).astype('float64')
    
                    # one hot encode RBPs
                    if np.all(RBPs == '') == False:
                        num_RBPs = np.array(len(RBPs), dtype=np.int64)
                        RBP_onehot = np.zeros((len_DNA,num_RBPs), dtype=np.int8)
                        for i, rbp in enumerate(RBPs):
                            p = np.array(peaks[rbp].loc[gene], dtype=int)
                            p = p - int(seq_start_)
                            p[p<0] = 0
                            p[p>len_DNA] = 0
        
                            for j in p:
                                RBP_onehot[j[0]:j[1],i] = 1
        
                        # make example
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'length': _bytes_feature(seq_len.flatten().tostring()),
                            'numRBPs': _bytes_feature(num_RBPs.flatten().tostring()),
                            'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                            'target': _bytes_feature(targets.flatten().tostring()),
                            'splicing': _bytes_feature(splicing.flatten().tostring()),
                            'peaks': _bytes_feature(RBP_onehot.flatten().tostring())}))
                    else:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'length': _bytes_feature(seq_len.flatten().tostring()),
                            'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                            'target': _bytes_feature(targets.flatten().tostring()),
                            'splicing': _bytes_feature(splicing.flatten().tostring())}))
    
                    # write
                    writer.write(example.SerializeToString())
    
    #                 # advance indexes
                    ti += 1
                    si += 1
    
    fasta_open.close()
    
    # Write statistics.json
    
    stats_dict = {}
    stats_dict['num_targets'] = 1
    stats_dict['seq_length'] = max_seq_length
    stats_dict['target_length'] = 1
    
    for fi in range(num_folds):
        stats_dict['%s_seqs' % fold_labels[fi]] = len(fold_indexes[fi])
    
    with open('%s/statistics.json' % fold_dir, 'w') as stats_json_out:
        json.dump(stats_dict, stats_json_out, indent=4)
    
    # Copy the params.json
    train_dict = {}
    train_dict['batch_size'] = 64
    train_dict['optimizer'] = 'adam'
    train_dict['loss'] = 'mse'
    train_dict['learning_rate'] = 0.0001
    train_dict['adam_beta1'] = 0.90
    train_dict['adam_beta2'] = 0.998
    train_dict['global_clipnorm'] = 0.5
    train_dict['train_epochs_min'] = 100
    train_dict['train_epochs_max'] = 500
    train_dict['patience'] = 50
    
    model_dict = {}
    model_dict['activation'] = 'relu'
    model_dict['rnn_type'] = 'gru'
    model_dict['seq_length'] = max_seq_length
    if np.all(RBPs == '') == False:
        model_dict['seq_depth'] = len(RBPs) + 5
    else:
        model_dict['seq_depth'] = 5
    model_dict['augment_shift'] = 3
    model_dict['num_targets'] = 1
    model_dict['heads'] = 1
    model_dict['filters'] = 64
    model_dict['kernel_size'] = 5
    model_dict['dropout'] = 0.3
    model_dict['l2_scale'] = 0.001
    model_dict['ln_epsilon'] = 0.007
    model_dict['num_layers'] = num_layers
    model_dict['bn_momentum'] = 0.90
    
    params_dict = {}
    params_dict['train'] = train_dict
    params_dict['model'] = model_dict
    
    with open('%s/params.json' % fold_dir, 'w') as params_json_out:
        json.dump(params_dict, params_json_out, indent=4)


# Function to generate the data
def create_multihead_dir(fold, general_out_dir, max_length, num_layers):
    
    fold_dir = general_out_dir + '/fold' + str(num_fold)
    Path(fold_dir).mkdir(parents=True, exist_ok=True)

    # # Write statistics.json
    # stats_dict = {}
    # stats_dict['num_targets'] = 2
    # stats_dict['seq_length'] = max_seq_length
    # stats_dict['target_length'] = 1
    
    # for fi in range(num_folds):
    #     stats_dict['%s_seqs' % fold_labels[fi]] = len(fold_indexes[fi])
    
    # with open('%s/statistics.json' % fold_dir, 'w') as stats_json_out:
    #     json.dump(stats_dict, stats_json_out, indent=4)
    
    max_seq_length = max_length
    
    # Copy the params.json
    train_dict = {}
    train_dict['batch_size'] = 64
    train_dict['optimizer'] = 'adam'
    train_dict['loss'] = 'mse'
    train_dict['learning_rate'] = 0.0001
    train_dict['adam_beta1'] = 0.90
    train_dict['adam_beta2'] = 0.998
    train_dict['global_clipnorm'] = 0.5
    train_dict['train_epochs_min'] = 100
    train_dict['train_epochs_max'] = 500
    train_dict['patience'] = 50
    
    model_dict = {}
    model_dict['activation'] = 'relu'
    model_dict['rnn_type'] = 'gru'
    model_dict['seq_length'] = max_seq_length
    if np.all(RBPs == '') == False:
        model_dict['seq_depth'] = len(RBPs) + 5
    else:
        model_dict['seq_depth'] = 5
    model_dict['augment_shift'] = 3
    model_dict['num_targets'] = [1,1]
    model_dict['heads'] = 2
    model_dict['filters'] = 64
    model_dict['kernel_size'] = 5
    model_dict['dropout'] = 0.3
    model_dict['l2_scale'] = 0.001
    model_dict['ln_epsilon'] = 0.007
    model_dict['num_layers'] = num_layers
    model_dict['bn_momentum'] = 0.90
    
    params_dict = {}
    params_dict['train'] = train_dict
    params_dict['model'] = model_dict
    
    with open('%s/params.json' % fold_dir, 'w') as params_json_out:
        json.dump(params_dict, params_json_out, indent=4)


# Do the CV
genes = exon_info['ENSG']
cv = GroupKFold(n_splits=10)
num_fold = 0

# for train_val_idxs, test_idxs in cv.split(PSI_glia, PSI_glia, genes):  
for i in range(10):
    test_idxs = exon_folds['Fold'] == i
    train_val_idxs = exon_folds['Fold'] != i
    
    cv2 = GroupKFold(n_splits=9)
    train_val_indices = list(cv2.split(PSI_glia.iloc[train_val_idxs],
                                       PSI_glia.iloc[train_val_idxs],
                                       genes[train_val_idxs]))
    train_idxs, val_idxs = train_val_indices[0]
    train_idxs = train_val_idxs[train_idxs]
    val_idxs = train_val_idxs[val_idxs]
        
    print('Size training set: ')
    print(len(train_idxs))
    print('Size validation set: ')
    print(len(val_idxs))
    print('Size test set: ')
    print(len(test_idxs))
        
    if var_threshold > 0:
        idx_var = np.where(np.abs(PSI_glia-PSI_neur) > var_threshold)[0]
        train_idxs = np.intersect1d(train_idxs, idx_var)
        val_idxs = np.intersect1d(val_idxs, idx_var)
        test_idxs = np.intersect1d(test_idxs, idx_var)
    
        print('\n')
        print('After filtering: ')
        print('Size training set: ')
        print(len(train_idxs))
        print('Size validation set: ')
        print(len(val_idxs))
        print('Size test set: ')
        print(len(test_idxs))
        
    if exon_list != 0:
        
        train_idxs = train_idxs[np.isin(genes[train_idxs], exons_tokeep)]
        val_idxs = val_idxs[np.isin(genes[val_idxs], exons_tokeep)]
        test_idxs = test_idxs[np.isin(genes[test_idxs], exons_tokeep)]
        
    
    if (cell_type == 'glia') or (cell_type == 'both'):
        create_tfrecords(PSI_glia, exon_info, train_idxs,
                         val_idxs, test_idxs, num_fold, 
                         general_out_dir + '/glia', peaks,
                         max_length, num_layers, padding, encode)
        
    if (cell_type == 'neurons') or (cell_type == 'both'):
        create_tfrecords(PSI_neur, exon_info, train_idxs,
                         val_idxs, test_idxs, num_fold, 
                         general_out_dir + '/neur', peaks,
                         max_length, num_layers, padding, encode)
        
    if (cell_type == 'both'):
        create_multihead_dir(num_fold, general_out_dir + '/both',
                             max_length, num_layers)
    
    
    num_fold += 1
    if num_fold == 10:
        break
    
    
