# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:03:57 2022

@author: lcmmichielsen
"""

import os
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import pysam

import tensorflow as tf

os.chdir('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/saluki')

from basenji.dna_io import dna_1hot


##### DONT USE THIS SCRIPT ANYMORE

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dir',            dest='dir',             default='/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/Data/HippData',
                    help='Directory to with PSI values')
parser.add_argument('--out',            dest='out',             default='tfrecords_0_both',
                    help='Output directory for the TFRecords')
parser.add_argument('--var_threshold',     dest='var_threshold',          type=float,     default=0.0,
                    help='Filter exons based on minimum delta PSI between neurons and glia')
parser.add_argument('--cell_type',      dest='cell_type',           type=str,       default='both',
                    help='Which celltype') # 'glia' / 'neurons' / 'both'
args = parser.parse_args()

general_dir = args.dir
out_dir = args.out
cell_type = args.cell_type
var_threshold = args.var_threshold

tfr_dir = general_dir + '/' + out_dir

try:
    os.mkdir(tfr_dir)
except:
    pass

# Read both the neuronal and glia PSI values
# We need both for var threshold, even if we're interested in single-task model
PSI_glia = pd.read_csv(general_dir + '/PSI_glia_norm.csv', index_col = 0)
PSI_neur = pd.read_csv(general_dir + '/PSI_neur_norm.csv', index_col = 0)

idx_tokeep = (PSI_neur['0'] >= 0) & (PSI_glia['0'] >= 0)
PSI_neur = PSI_neur.loc[idx_tokeep]
PSI_glia = PSI_glia.loc[idx_tokeep]

# Characteristics of the exons (needed to extract seq. later on)
exon_info = pd.DataFrame(PSI_glia.index)[0].str.split('_', expand=True)
exon_info = exon_info.rename(columns={0: 'chr', 1: 'start', 2: 'end',
                                      3: 'ENSG', 4: 'strand'})

# PSI_glia = np.mean(PSI.iloc[:,2:4], axis=1)
# PSI_neurons = np.mean(PSI.iloc[:,4:6], axis=1)
# PSI_glia_neurons = pd.concat((PSI_glia, PSI_neurons),axis=1)
# PSI_glia_neurons.columns = ['Glia', 'Neurons']
# PSI_glia_neurons['varStatus'] = PSI['variabilityStatus']
# PSI_glia_neurons.index = PSI.index
# PSI_glia_neurons['HumanGene'] = PSI['HumanGene']
# PSI_tokeep = PSI_glia_neurons[np.sum(PSI_glia_neurons.isna(), axis=1) == 0]


# Do the CV
# Now we only save 1 fold to test things quickly
# In the future, save all folds
genes = exon_info['ENSG']
fold=0
cv = GroupKFold(n_splits=10)

### Look at the RBP code to see how we did it for all folds!!
for train_val_idxs, test_idxs in cv.split(PSI_glia, PSI_glia, genes):   
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
    
    # Filter the train-val-test idxs by var status
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
    
    break

fold_indexes = [train_idxs, val_idxs, test_idxs]


# Write the file genes.tsv
split = np.zeros((len(exon_info),1), dtype='<U5')
split[train_idxs] = 'train'
split[val_idxs] = 'valid'
split[test_idxs] = 'test'

target = PSI_tokeep.values[:,:2]
genes = pd.DataFrame(np.hstack((split,np.reshape(PSI_tokeep['variabilityStatus'].values,(-1,1)),target)), 
                     index=PSI_tokeep.index, columns=np.hstack((['split', 'varStatus'], PSI_tokeep.columns[:2])))
genes = genes[genes['split'] != '']
genes.to_csv(general_dir + '/genes.csv')

# Options TFR writer
# To Do: make these options of the argparser..
tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

seqs_per_tfr = 256
fold_labels = ['train', 'valid', 'test']
max_seq_length = 3072
padding = 400

# open FASTA
fasta_file = '../Ref/GRCh38.primary_assembly.genome.fa'
fasta_open = pysam.Fastafile(fasta_file)

def rc(seq):
    return seq.translate(str.maketrans("ATCGatcg","TAGCtagc"))[::-1]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Write the TFRecords
num_folds = 3 #train-val-test

for fi in range(num_folds):
    exon_ID_set = PSI_tokeep.index[fold_indexes[fi]]
    exon_info_set = exon_info.iloc[fold_indexes[fi]]
    PSI_val_set = PSI_tokeep.iloc[fold_indexes[fi],:1]
    
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
                gene = exon_ID_set[si]
                seq_chrm = exon_info_set['chr'].iloc[si]
                seq_start = int(exon_info_set['start'].iloc[si])
                seq_end = int(exon_info_set['end'].iloc[si])+1 #Because fasta.fetch doesn't include end bp
                seq_strand = exon_info_set['strand'].iloc[si]
                
                if(seq_end-seq_start+2*padding) > max_seq_length:
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
                assert(len(seq_dna) <= max_seq_length)

                # orient
                if seq_strand == '-':
                    seq_dna = rc(seq_dna)

                # one hot code
                seq_1hot = dna_1hot(seq_dna)
                seq_len = np.array(len(seq_dna), dtype=np.int64)
                
                # splicing
                splicing = np.zeros((seq_len,1), dtype=np.int8)
                splicing[splicing_ind] = 1

                # get targets
                targets = PSI_val_set.iloc[si].values
                targets = targets.reshape((1,-1)).astype('float64')
                
                # make example
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
stats_dict['num_targets'] = PSI_tokeep.shape[1]-2
stats_dict['seq_length'] = max_seq_length
stats_dict['target_length'] = 1

for fi in range(num_folds):
    stats_dict['%s_seqs' % fold_labels[fi]] = len(fold_indexes[fi])

with open('%s/statistics.json' % general_dir, 'w') as stats_json_out:
    json.dump(stats_dict, stats_json_out, indent=4)

# Copy the params.json

shutil.copy2('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/HumanHipp/tfrecords_glia_neurons/params.json', general_dir)
