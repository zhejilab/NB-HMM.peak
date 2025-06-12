import sys
import os
import pickle
import argparse

import numpy as np
import random
from statsmodels.stats import multitest 
import hmm_custom

############################################################
#                                                          #
#                     input paramerters                    #
#                                                          #
############################################################

parser = argparse.ArgumentParser(description="Peak Calling from Sequencing Data at Nucleotide Resolution Using a Negative Binomial Hidden Markov Model (HMM)")

parser.add_argument("-s", required=True, help="Path to the sense strand BED file")
parser.add_argument("-a", required=True, help="Path to the antisense strand BED file")
parser.add_argument("-o", required=True, help="Path to the output directory")
parser.add_argument("-t", type=int, default=1, help="Fraction of data used for HMM training (default: 1, i.e., 100%)")

args = parser.parse_args()

############################################################
#                                                          #
#                        functions                         #
#                                                          #
############################################################

def read_in_bed(sense_bed, antisense_bed, sampled_fraction):
    
    sense_peaks = {}
    antisense_peaks = {}

    with open(sense_bed, "r") as r:
        for line in r.readlines():
            chrom, pos_minus1, pos, value = line.rstrip().split("\t")
            sense_peaks[chrom+":+:"+str(pos)] = float(value)

    with open(antisense_bed, "r") as r:
        for line in r.readlines():
            chrom, pos_minus1, pos, value = line.rstrip().split("\t")
            antisense_peaks[chrom+":-:"+str(pos)] = float(value)

    peaks = {**sense_peaks, **antisense_peaks}
    sampled_keys = random.sample(list(peaks.keys()), int(len(peaks) * sampled_fraction))
    sampled_peaks = {k: peaks[k] for k in sampled_keys}

    return sense_peaks, antisense_peaks, sampled_peaks


def hmm_training(sampled_peaks, out_file_path):

    X = np.array(list(sampled_peaks.values())).reshape(-1,1)
    model = hmm_custom.NegativeBinomialHMM(n_components=2)
    model = model.fit(X)

    with open(out_file_path+"/hmm.pkl", "wb") as f: 
        pickle.dump(model, f)

    f.close()


def load_hmm_pkl(pkl_path):
    picklefile = open(pkl_path,'rb')
    model = pickle.load(picklefile)
    picklefile.close()
    return model


def hmm_prediction(peaks, strand, out_file_path):

    model = load_hmm_pkl(out_file_path+"/hmm.pkl")
    
    X = np.array(list(peaks.values())).reshape(-1,1)
    l,p = model.score_samples(X)
    p_a = np.array(p)
    p_values = p_a[:,1]

    w1 = open(out_file_path + "/hmm_p." + strand + ".bed", "w") 
    for i in range(len(peaks.keys())):
        chrom, strand, pos = list(peaks.keys())[i].split(":")
        pos = int(pos)
        w1.write(chrom+"\t"+str(pos-1)+"\t"+str(pos)+"\t"+str(p_values[i])+"\n")
    w1.close()

############################################################
#                                                          #
#                         Results                          #
#                                                          #
############################################################

sense_peaks, antisense_peaks, training_set = read_in_bed(args.s, args.a, args.t)

hmm_training(training_set, args.o)

hmm_prediction(sense_peaks, "+", args.o)
hmm_prediction(antisense_peaks, "-", args.o)
