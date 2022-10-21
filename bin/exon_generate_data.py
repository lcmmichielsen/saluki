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
from sklearn.model_selection import StratifiedGroupKFold

import pysam

import tensorflow as tf

os.chdir('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/saluki')

from basenji.dna_io import dna_1hot

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dir',            dest='dir',             default='/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/HumanHipp/tfrecords_glia_neurons',
                    help='Directory with the sequences and half-life time')
parser.add_argument('--var_tokeep',     dest='var_tokeep',          type=str,       default="Cons,High,Low,Medium",
                    help='Which type of exons to use during training')
args = parser.parse_args()

general_dir = args.dir
var_tokeep = np.asarray(args.var_tokeep.split(','))

try:
    os.mkdir(general_dir)
except:
    pass

tfr_dir = general_dir + '/tfrecords'

try:
    os.mkdir(tfr_dir)
except:
    pass

# We focus now on the PSI values for Neurons & Glia
# Make the cell type of interest an argument in the future
PSI_var = pd.read_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/HumanHipp/altInHumans_10_90_cellType_withClassification', sep='\t')
PSI_cons = pd.read_csv('/tudelft.net/staff-bulk/ewi/insy/DBL/lmichielsen/PSI_project/HumanHipp/consInHumans_5_95_cellType', sep='\t')
PSI_cons['variabilityStatus'] = 'Cons'
PSI = pd.concat((PSI_var, PSI_cons))
PSI.index = PSI['HumanExon'] + '_' + PSI['HumanGene']

PSI_glia = np.mean(PSI.iloc[:,2:4], axis=1)
PSI_neurons = np.mean(PSI.iloc[:,4:6], axis=1)

PSI_glia_neurons = pd.concat((PSI_glia, PSI_neurons),axis=1)
PSI_glia_neurons.columns = ['Glia', 'Neurons']
PSI_glia_neurons['varStatus'] = PSI['variabilityStatus']
PSI_glia_neurons.index = PSI.index
PSI_glia_neurons['HumanGene'] = PSI['HumanGene']

PSI_tokeep = PSI_glia_neurons[np.sum(PSI_glia_neurons.isna(), axis=1) == 0]


# Do the CV
# Now we only save 1 fold to test things quickly
# In the future, save all folds
genes = PSI_tokeep['HumanGene']
varStatus = PSI_tokeep['varStatus']
fold=0
cv = StratifiedGroupKFold(n_splits=10, random_state=0, shuffle=True)

for train_val_idxs, test_idxs in cv.split(varStatus, varStatus, genes):   
    cv = StratifiedGroupKFold(n_splits=9, random_state=0, shuffle=True)
    ynew = varStatus[train_val_idxs]
    groupsnew = genes[train_val_idxs]
    for train_idxs, val_idxs in cv.split(ynew, ynew, groupsnew):
        train_idxs = train_val_idxs[train_idxs]
        val_idxs = train_val_idxs[val_idxs]
        break
    
    print('Size training set: ')
    print(len(train_idxs))
    print('Size validation set: ')
    print(len(val_idxs))
    print('Size test set: ')
    print(len(test_idxs))
    
    # Filter the train-val-test idxs by var status
    train_idxs = train_idxs[np.squeeze(np.isin(varStatus[train_idxs], var_tokeep))]
    val_idxs = val_idxs[np.squeeze(np.isin(varStatus[val_idxs], var_tokeep))]
    test_idxs = test_idxs[np.squeeze(np.isin(varStatus[test_idxs], var_tokeep))]
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

# Characteristics of the exons (needed to extract seq. later on)
exon_info = pd.DataFrame(data=np.zeros((len(PSI_tokeep), 5)), 
                          columns=['chr', 'start', 'end', 'ENSG', 'strand'])

for i in range(len(PSI_tokeep)):
    
    exon_info.iloc[i, [0,1,2,4,3]] = PSI_tokeep.index[i].split('_')

exon_info.iloc[:5]

# Write the file genes.tsv
split = np.zeros((len(exon_info),1), dtype='<U5')
split[train_idxs] = 'train'
split[val_idxs] = 'valid'
split[test_idxs] = 'test'

target = PSI_tokeep.values[:,:2]
genes = pd.DataFrame(np.hstack((split,np.reshape(PSI_tokeep['varStatus'].values,(-1,1)),target)), 
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
    PSI_val_set = PSI_tokeep.iloc[fold_indexes[fi],:2]
    
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
