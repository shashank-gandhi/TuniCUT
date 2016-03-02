"""
Author: Shashank Gandhi
GitHub: shashank357
Email: shashank.gandhi@caltech.edu
"""
#import in-built modules
from optparse import OptionParser
import re
import sqlite3
import itertools

#import biopython modules
from Bio.Seq import Seq
from Bio import SeqIO

#import machine learning modules
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model

#import scipy modules
import numpy as np
import pandas as pd
import time

start = time.time()

#initiate parser
parser=OptionParser()

#filename handle for sequences
parser.add_option("-f", "--fasta", action="store", type="string", dest="fasta_filename", help="File containing FASTA sequences. Please ensure that each sequence has a unique identifier. Check tutorial for examples.")

#handle for training data
parser.add_option("-t", "--training_data", action = "store", type="string", dest="training_data", help="File containing training data for machine learning. Location: 'TuniCUT/TrainingData.txt' ")

#handle for complete data
parser.add_option("-c", "--complete_data", action="store", type="string", dest="complete_data", help="File containing complete dataset from high throughput sequencing. Location: 'TuniCUT/CompleteData.csv' ")

parser.add_option("-o", "--output", action="store", type="string", dest="output", help="Text file that will store the output. Please ensure that the filename contains a '.txt' extension. Example: 'Predictions_feb11_output.txt' ")

#handle for gff_file:This is a new addition to v3.0.
parser.add_option("-g", "--gff_filename", action="store", type="string", dest="gff_filename", help="Optional handle for output gff3 file. This can be loaded into IGV to directly visualize sequences. If no name is given, the name of the gff3 file will be used with a '.gff3' extension. Example: Predictions_feb11.gff3")

(options, args) = parser.parse_args()

#make sure all options have been accounted for. Give names to options for which the user hasn't provided any information.

#check output file for extension
if not options.output[-4:]=='.txt':
    options.output = options.output + ".txt"

#create gff filename if missing
if not options.gff_filename:
    options.gff_filename = options.output[:-4]+".gff3"

#print information for the user

print("Now looking at ",options.fasta_filename, " using ", options.training_data, " and ", options.complete_data, " to write output in ", options.output, ".\n", options.gff_filename, "gff file has also been created in your working directory. You can load these files on IGV.\n\nThis script will run 28 blocks of code. Completion can be tracked using printed statements on the screen.")

def ReadFasta(fasta_file):
    """
    This function will allow the user to read a Fasta file and store all sequences as a dictionary using fasta header as keys.
    """

    print ("1. Opening {}.......".format(fasta_file))

    print ("2. Reading {}.......".format(fasta_file))

    #open and read the fasta file
    with open(fasta_file) as in_handle:
        record_dict = SeqIO.to_dict(SeqIO.parse(in_handle, "fasta"))

    print("3. Creating FastaSequence Dictionary")
    #create empty dictionary
    FastaSequence = {}

    for key in record_dict.keys():
        FastaSequence[key] = str(record_dict[key].seq).upper()

    return(FastaSequence)

Fastas = ReadFasta(options.fasta_filename)

print ("4. Created Fastas dictionary")

AllguideRNAs = []

print ("5. Entering findgRNA function")

def findgRNA(ScaffoldDict):
    """
    This function takes a dictionary of sequences (similar to one created in previous function) and finds all "NGG" and "CCN" gRNAs.
    """

    for scaffoldsID in ScaffoldDict.keys():

        sequence=ScaffoldDict[scaffoldsID] #store sequence

        TemplateSites=re.findall( r'(?=([ATGC]{20}GG))', sequence )   #NGG guideRNAs

        NonTemplateSites=re.findall( r'(?=(CC[ATGC]{20}))', sequence )   #CCN guideRNAs

        TemplateSitesPositions=[ int( num.start() + 1) for num in re.finditer( r'(?=([ATGC]{20}GG))', sequence )  ] #NGG guideRNAs positions

        NonTemplateSitesPositions=[ int( num.start() + 1) for num in re.finditer( r'(?=(CC[ATGC]{20}))', sequence ) ]  #CCN guideRNAs positions

        sequence43bpT = [sequence[i-12:i+31] for i in TemplateSitesPositions]
        #replace first nt in the guide to "G"
        sequence43bpTcorrected = [str(i[:10]+"G"+i[11:]) for i in sequence43bpT]
        #CCN guides
        sequence43bpNT = [sequence[i-11:i+32] for i in NonTemplateSitesPositions]
        #replace first 32nt with "C"
        sequence43bpNTcorrected = [str(Seq(str(i[:32]+"C"+i[33:])).reverse_complement()) for i in sequence43bpNT]
        #take reverse complement
        TemplatePAMs = [i[30:33] for i in sequence43bpTcorrected]

        NonTemplatePAMs = [i[30:33] for i in sequence43bpNTcorrected]

        for num in range(len(TemplateSites)):

            if (not re.search(r'(?=TTT)', TemplateSites[num]) and not re.search(r'(?=N)', sequence43bpTcorrected[num]) ):

                AllguideRNAs.append( (scaffoldsID, "+", TemplatePAMs[num], TemplateSitesPositions[num] , TemplateSitesPositions[num]+18, sequence43bpTcorrected[num], sequence43bpTcorrected[num][10:30],TemplateSitesPositions[num]+19))

        for num in range(len(NonTemplateSites)):

            if ( not re.search(r'(?=AAA)', NonTemplateSites[num]) and not re.search(r'(?=N)', sequence43bpNTcorrected[num]) ):

                AllguideRNAs.append( (scaffoldsID, "-", str(Seq(str(NonTemplatePAMs[num])).reverse_complement()), NonTemplateSitesPositions[num]+3 , NonTemplateSitesPositions[num]+21, sequence43bpNTcorrected[num], sequence43bpNTcorrected[num][10:30],NonTemplateSitesPositions[num]-1 ))

findgRNA(Fastas)

print ("6. Created AllguideRNAs list. Now adding 43bp length filter")

AllguideRNAs = [i for i in AllguideRNAs if len(i[5])==43]

print ("\nThe total number of guide RNAs in your file is", len(AllguideRNAs))

print ("7. Filter applied. Now sorting AllguideRNAs list")
AllguideRNAs = sorted( AllguideRNAs, key= lambda x: x[3])

print ("8. AllguideRNAs list sorted")

#score predictions
TestSequencesForPredictions = [ str(tup[5]) for tup in AllguideRNAs]

print("9. sequences separated for machine learning")

print ("10. Entering machine learning block")
#machine learning

complete_data = pd.read_csv(options.complete_data)
complete_data = complete_data.drop(complete_data.index[77:89])

print ("11. read", options.complete_data," file.")
#read in sequences and scores

with open(options.training_data, "r") as handle:
    guides = [i.strip().split("\t") for i in handle.readlines()[1:]]

#sequences are 43nt long
sequences = [str(i[3]).upper() for i in guides]+TestSequencesForPredictions

print ("12. Created sequences for machine learning")

#number of continuous response variables. can be used to add variables later on.
num_continuous = 0

#changing response variable to categorical
scores=list(complete_data.Mutagenesis)
print ("13. Separated scores")

#guides 19nt long sequences
gRNA = [i[11:30] for i in sequences]

print ("14. separated guideRNAs")

#PAM-proximal and distal nucleotides
pampn = [i[13:] for i in gRNA]

#g and c content of the PAM proximal nucleotides
pampn_g = [str(int(i.count("G")*100/len(i))) for i in pampn]
pampn_c = [str(int(i.count("C")*100/len(i))) for i in pampn]

print ("15. Calculated PAMPN G and C percentages.")

#label categorical variables
replace_dict = {"A":"0","T":"1","G":"2","C":"3",\
"AA":"4", "AT":"5","AG":"6","AC":"7",\
"TA":"8","TT":"9","TG":"10","TC":"11","GA":"12","GT":"13",\
"GG":"14","GC":"15","CA":"16","CT":"17","CG":"18","CC":"19"}

g_content_dict = {"0":"20","16":"21","33":"22","50":"23","66":"24","83":"25","100":"26"}
c_content_dict = {"0":"27","16":"28","33":"29","50":"30","66":"31","83":"32","100":"33"}

#if you want to add more categorical parameters, define a dictionary that defined the variable (34 onwards), create a list with these variables (see above), and call the replace_dinucleotide funciton.

def replace_dinucleotide(list_name, replace_what, with_what):
    """
    Takes a list as input, uses pop and insert methods to replace
    specific nucleotide/di-nucleotide with a categorical number defined
    in the previous step in replace_dict.
    """

    list_name[np.where(list_name == replace_what)] = with_what

    return list_name

print ("16. Replacing G and C features with appropriate binary code.")

#replace gc-content by categorical feature def
pampn_g = np.array(pampn_g)
for i in g_content_dict.keys():
    pampn_g = replace_dinucleotide(pampn_g, i, g_content_dict[i])

#make sure variables are integers.
pampn_g = pampn_g.astype(np.int)

#create two dimensional numpy array for faster calculations.
pampn_g = np.reshape(pampn_g, (len(pampn_g),1))

print ("17. binary code replaced for G")

pampn_c = np.array(pampn_c)
for i in c_content_dict.keys():
    pampn_c = replace_dinucleotide(pampn_c, i, c_content_dict[i])

#make sure variables are integers
pampn_c = pampn_c.astype(np.int)

#create 2 dimentional numpy array for faster calculations
pampn_c = np.reshape(pampn_c, (len(pampn_c),1))

print ("18. binary code replaced for C")

sequences_combinations=[]
for i in sequences:
    sequences_combinations.append(list(i))

#define di-nucleotide features
all_features = []

for k in range(len(sequences)):
    #itertools will create all possible dual-nucleotide combinations. (43C2)

    all_features.append(sequences_combinations[k]+[''.join(j) for j in itertools.combinations(sequences[k], 2)])

no_change_all_features = all_features

#all_features now contains both single and di-nucleotide features

all_features = np.array(all_features)

for i in range(len(all_features)):
    for key in replace_dict.keys():
        output = replace_dinucleotide(all_features[i], key, replace_dict[key])
        all_features[i] = output
all_features = all_features.astype(np.int)

seq_list=[]

#add more features by just concatinating using the step below.
SequenceData = np.concatenate((all_features, pampn_g, pampn_c), axis=1)

print ("77 sequences have been added for machine learning. Now looking at", SequenceData.shape[1], " features.")

ScoresData = np.array(complete_data.Mutagenesis)

#number of times each feature is present in the dataset
max_occ = np.empty(SequenceData.shape[1]-num_continuous, dtype=int)

for i in range(SequenceData.shape[1]-num_continuous):
    max_occ[i] = len(set(SequenceData[:,i]))

#occupancy of each feature in the data given as an input
occupancy={}
for i in range(SequenceData.shape[1]-num_continuous):
    occupancy[i] = list(set(SequenceData[:,i]))

#use this masker when using continuous variables and categorical variables together.
mask_list = [True]*(SequenceData.shape[1]-num_continuous) + [False]*num_continuous

#setup encoder
print ("19. Setting up one hot encoder")
enc = OneHotEncoder(categorical_features=np.array(mask_list))

#save one-hot encoding as array
print ("20. transforming one hot encoded array")
enc_transformed_array = enc.fit_transform(SequenceData).toarray()

test_these_seq = enc_transformed_array[77:,:]
enc_transformed_array = enc_transformed_array[:77,:]

#if you want to use top and bottom 25% data for test/train split:

print ("21. Separated training data")
training_data = np.array(list(enc_transformed_array[:28,:]) + list(enc_transformed_array[47:77,:]))

test_data = np.array(list(test_these_seq))

training_scores = np.array(list(scores)[:28] + list(scores)[47:77])

#cross validation to get best value for alpha, cannot do this here because the model will start overfitting. Not enough data points to perform bootstrapping.

print ("22. Fitting model to training dataset")
clf = linear_model.Lasso(fit_intercept = True, alpha=0.005, max_iter=100000)

#fit the model
clf.fit(training_data, training_scores)

print ("23. Predicting scores for all sequences")
#predict scores
pred_scores = clf.predict(test_data)

pred_scores [ pred_scores < 0] = 0

print ("shape of pred_scores is",pred_scores.shape)

print ("24. Created output list WholeGenomeGuidesWithPredictedScores")
WholeGenomeGuidesWithPredictedScores = []

print ("25. appending information to WholeGenomeGuidesWithPredictedScores list")
for i in range(len(AllguideRNAs)):

    WholeGenomeGuidesWithPredictedScores.append( [AllguideRNAs[i][0], ".", AllguideRNAs[i][2], AllguideRNAs[i][3], AllguideRNAs[i][4], ".", AllguideRNAs[i][1], ".", "sgRNA: " + AllguideRNAs[i][6]+ " Score:"+str(pred_scores[i])+" OSOPCR-Fwd-Primer: "+str(AllguideRNAs[i][6])+"GTTTAAGAGCTATGCTGGAAACAG; OSOPCR-Rev-Primer: "\
    + str(Seq(AllguideRNAs[i][6]).reverse_complement())+"atctataccatcggatgccttc"])

print ("26. writing ", options.output, " file")

with open(options.output, "w+") as output_handle:
    header = "FastaHeader\tPAM\tStart\tEnd\tsgRNA\tScore\tOSOPCR-Fwd-Primer\tOSOPCR-Rev-Primer\n"
    output_handle.write(header)
    for i in range(len(AllguideRNAs)):
#    for tuples in WholeGenomeGuidesWithPredictedScores:
        temp = str(AllguideRNAs[i][0])+"\t"+str(AllguideRNAs[i][2])+"\t"+str(AllguideRNAs[i][3])+"\t"+ \
        str(AllguideRNAs[i][4])+"\t"+str(AllguideRNAs[i][6])+"\t"+str(pred_scores[i])+ \
        "\t"+str(AllguideRNAs[i][6])+"GTTTAAGAGCTATGCTGGAAACAG"+"\t"+str(Seq(AllguideRNAs[i][6]).reverse_complement())+\
        "atctataccatcggatgccttc"+"\n"

        output_handle.write(temp)

output_handle.close()

print ("27. writing gff3 file for output")
with open(options.gff_filename, "w+") as WriteHandle:

    for tuples in WholeGenomeGuidesWithPredictedScores:
        temp = str(tuples[0])+"\t"+str(tuples[1])+"\t"+str(tuples[2])+"\t"+ \
        str(tuples[3])+"\t"+str(tuples[4])+"\t"+str(tuples[5])+"\t"+ \
        str(tuples[6])+"\t"+str(tuples[7])+ "\t" + str(tuples[8])+ "\n"

        WriteHandle.write(temp)

print ("28. GFF3 file written, closing handle.")
WriteHandle.close()

with open(options.gff_filename+".bedgraph", "w+") as AnotherWriteHandle:
    AnotherWriteHandle.write("""track type=bedGraph name=TuniCUT predictions description = Predicted Mutagenesis Rates by TuniCUT useScore=1\n""")
    for i in range(len(WholeGenomeGuidesWithPredictedScores)):
        temp =str(WholeGenomeGuidesWithPredictedScores[i][0])+"\t"+str(AllguideRNAs[i][7])+"\t"+str(int(AllguideRNAs[i][7])+3)+"\t"+str(pred_scores[i])+"\n"

        AnotherWriteHandle.write(temp)

print ("29. Written bedgraph file.")
AnotherWriteHandle.close()

print ("Done.")

print('The script took ', time.time()-start, ' seconds.')
