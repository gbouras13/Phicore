import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from typing import List, Union
import os
import sys
import gzip
from Bio import SeqIO
import binascii


__author__ = 'Przemyslaw Decewicz; George Bouras'

# Przemyslaw Decewicz
def get_features_of_type(seqiorec: SeqRecord, ftype: str) -> List[SeqFeature]:
    """
    Get features of a given type from SeqRecord
    :param seqiorec: a SeqRecord object
    :param ftype: type of a feature
    :return:
    """

    flist = []
    for feature in seqiorec.features:
        if feature.type == ftype:
            flist.append(feature)
    
    return flist

def get_gc_content(seq: Union[str, Seq, SeqRecord]) -> float:
    """
    Calculate GC content of a nucleotide sequence
    :param seq: a nucleotide sequence
    :return:
    """

    gc = 0
    for i in seq:
        if i == 'G' or i == 'C':
            gc += 1
    return gc / len(seq)

def get_features_lengths(seqiorec: SeqRecord, ftype: str) -> List[float]:
    """
    Get average length of SeqFeatures of a given type
    :param seqiorec: a SeqRecord object
    :param ftype: type of a feature
    :return:
    """

    lengths = []
    for feature in seqiorec.features:
        if feature.type == ftype:
            lengths.append(float(len(feature.location.extract(seqiorec).seq)))

    if ftype == 'CDS':
        return [x / 3 for x in lengths]
    else:
        return lengths

def get_coding_density(seqiorec: SeqRecord, ftypes: List[str] = ['CDS', 'tRNA', 'rRNA']) -> float:
    """
    Get coding density for a SeqRecord considering given features types
    :param seqiorec: SeqRecord object
    :param ftypes: a list of feature types
    :return:
    """

    cdcov = np.zeros(len(seqiorec.seq))
    for feature in seqiorec.features:
        if feature.type in ftypes:
            start, stop = map(int, sorted([feature.location.start, feature.location.end]))
            cdcov[start:stop] += 1
    return sum([1 if x > 0 else 0 for x in cdcov]) / len(seqiorec.seq)

def get_distribution_of_stops(seqiorec: SeqRecord, window: int = 210, step: int = 1) -> pd.DataFrame:
    """
    Get distribution of STOP codons in a sequence
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """

    stops = ['TAA', 'TAG', 'TGA']

    stops_distr = {
        'x': range(1, len(seqiorec.seq) + 1),
        'TAA': [np.NAN]*int(window/2),
        'TAG': [np.NAN]*int(window/2),
        'TGA': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        window_seq = seqiorec.seq[i : i + window]
        taa = window_seq.count('TAA')
        tag = window_seq.count('TAG')
        tga = window_seq.count('TGA')
        stops_distr['TAA'].extend([taa]*(step))
        stops_distr['TAG'].extend([tag]*(step))
        stops_distr['TGA'].extend([tga]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(stops_distr['TAA'])
    if left > 0:   
        stops_distr['TAA'].extend([np.NAN]*left)
        stops_distr['TAG'].extend([np.NAN]*left)
        stops_distr['TGA'].extend([np.NAN]*left)

    return pd.DataFrame(stops_distr)


# George Bouras
def get_mean_cds_length_rec_window(seqiorec : SeqRecord, window_begin : int, window_end : int) -> float:
    """
    Get mean CDS length
    :param seqiorec: SeqRecord object
    :param window_begin: integer
    :param window_end: integer
    :return:
    """

    cds_length = []
    for feature in seqiorec.features:
        if feature.type == 'CDS':
            if feature.location.start > window_begin and feature.location.start < window_end and feature.location.end > window_begin and feature.location.end < window_end:
                cds_length.append(len(feature.location.extract(seqiorec).seq)/3)
    if len(cds_length) == 0:
        mean = (window_end - window_begin)/3
    else:
        mean = np.mean(cds_length)
    return mean


def get_cds_count_length_rec_window(seqiorec : SeqRecord, window_begin : int, window_end : int) -> float:
    """
    Get median CDS length
    :param seqiorec: SeqRecord object
    :param window_begin: integer
    :param window_end: integer
    :return:
    """

    cds_length = []
    count = 0 
    for feature in seqiorec.features:
        if feature.type == 'CDS':
            if feature.location.start > window_begin and feature.location.start < window_end and feature.location.end > window_begin and feature.location.end < window_end:
                count +=1
    return count


def get_rolling_gc(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """

    gcs = ['G', 'C']

    gcs_distr = {
        'x': range(1, len(seqiorec.seq) + 1),
        'G': [np.NAN]*int(window/2),
        'C': [np.NAN]*int(window/2),
        'GC': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        window_seq = seqiorec.seq[i : i + window]
        g = window_seq.count('G')
        c = window_seq.count('C')
        gcs_distr['G'].extend([g]*(step))
        gcs_distr['C'].extend([c]*(step))
        gcs_distr['GC'].extend([g+c]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(gcs_distr['G'])
    if left > 0:   
        gcs_distr['G'].extend([np.NAN]*left)
        gcs_distr['C'].extend([np.NAN]*left)
        gcs_distr['GC'].extend([np.NAN]*left)

    return pd.DataFrame(gcs_distr)

def get_rolling_mean_cds(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """
    cds_average = {
        'x': range(1, len(seqiorec.seq) + 1),
        'Mean_CDS': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        cds_mean = get_mean_cds_length_rec_window(seqiorec,i, i + window )
        cds_average['Mean_CDS'].extend([cds_mean]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(cds_average['Mean_CDS'])
    if left > 0:   
        cds_average['Mean_CDS'].extend([np.NAN]*left)


    return pd.DataFrame(cds_average)


def get_rolling_count_cds(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """
    cds_count = {
        'x': range(1, len(seqiorec.seq) + 1),
        'Count_CDS': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        count = get_cds_count_length_rec_window(seqiorec,i, i + window )
        cds_count['Count_CDS'].extend([count]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(cds_count['Count_CDS'])
    if left > 0:   
        cds_count['Count_CDS'].extend([np.NAN]*left)


    return pd.DataFrame(cds_count)





def is_gzip_file(f):
    """
    This is an elegant solution to test whether a file is gzipped by reading the first two characters.
    I also use a version of this in fastq_pair if you want a C version :)
    See https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed for inspiration
    :param f: the file to test
    :return: True if the file is gzip compressed else false
    """
    with open(f, 'rb') as i:
        return binascii.hexlify(i.read(2)) == b'1f8b'



def parse_genbank(filename, verbose=False):
    """
    Parse a genbank file and return a Bio::Seq object
    """


    try:
        if is_gzip_file(filename):
            handle = gzip.open(filename, 'rt')
        else:
            handle = open(filename, 'r')
    except IOError as e:
        print(f"There was an error opening {filename}", file=sys.stderr)
        sys.exit(20)

    return SeqIO.parse(handle, "genbank")

def get_rolling_deltas(genbank_path_all : str, genbank_path_tag : str, genbank_path_tga : str, genbank_path_taa : str, window : int = 2000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param window: window size
    :param step: step size
    :return:
    """

    for record in parse_genbank(genbank_path_all):
        df_all = get_rolling_mean_cds(record, window=2000, step=30)
        count_all = get_rolling_count_cds(record, window=2000, step=30)
        #stops_all = get_distribution_of_stops(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tag):
        df_tag = get_rolling_mean_cds(record, window=2000, step=30)
        count_tag = get_rolling_count_cds(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tga):
        df_tga = get_rolling_mean_cds(record, window=2000, step=30)
        count_tga = get_rolling_count_cds(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_taa):
        df_taa = get_rolling_mean_cds(record, window=2000, step=30)
        count_taa = get_rolling_count_cds(record, window=2000, step=30)


    df_all['Mean_CDS_all'] = df_all['Mean_CDS'] 
    df_all['Mean_CDS_tag'] = df_tag['Mean_CDS'] 
    df_all['Mean_CDS_tga'] = df_tga['Mean_CDS'] 
    df_all['Mean_CDS_taa'] = df_taa['Mean_CDS'] 
    df_all['Count_CDS_all'] = count_all['Count_CDS'] 
    df_all['Count_CDS_tag'] = count_tag['Count_CDS'] 
    df_all['Count_CDS_tga'] = count_tga['Count_CDS'] 
    df_all['Count_CDS_taa'] = count_taa['Count_CDS'] 
    # df_all['TAG_all'] = stops_all['TAG'] 
    # df_all['TGA_all'] = stops_all['TGA'] 
    # df_all['TAA_all'] = stops_all['TAA'] 


    df_all['tag_minus_all_mean_cds'] = df_all['Mean_CDS_tag'] - df_all['Mean_CDS_all']
    df_all['tga_minus_all_mean_cds'] = df_all['Mean_CDS_tga'] - df_all['Mean_CDS_all']
    df_all['taa_minus_all_mean_cds'] = df_all['Mean_CDS_taa'] - df_all['Mean_CDS_all']
    df_all['tag_minus_all_count_cds'] = df_all['Count_CDS_tag'] - df_all['Count_CDS_all']
    df_all['tga_minus_all_count_cds'] = df_all['Count_CDS_tga'] - df_all['Count_CDS_all']
    df_all['taa_minus_all_count_cds'] = df_all['Count_CDS_taa'] - df_all['Count_CDS_all']

    df_all = df_all.drop(['Mean_CDS', 'Mean_CDS_all', 'Mean_CDS_tag', 'Mean_CDS_tga', 'Mean_CDS_taa', 'Count_CDS_all', 'Count_CDS_tag', 'Count_CDS_tga', 'Count_CDS_taa'], axis=1)

    return pd.DataFrame(df_all)



def get_rolling_stop_codon_usage(genbank_path_all : str, genbank_path_tag : str, genbank_path_tga : str, genbank_path_taa : str, window : int = 2000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param window: window size
    :param step: step size
    :return:
    """

    for record in parse_genbank(genbank_path_all):
        stops_all = get_distribution_of_stops(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tag):
        stops_tag = get_distribution_of_stops(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tga):
        stops_tga = get_distribution_of_stops(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_taa):
        stops_taa = get_distribution_of_stops(record, window=2000, step=30)


    stops_all['Count_Stop_TAG_all'] = stops_all['TAG'] 
    stops_all['Count_Stop_TGA_all'] = stops_all['TGA'] 
    stops_all['Count_Stop_TAA_all'] = stops_all['TAA'] 

    stops_all['Count_Stop_TAG_TAGrec'] = stops_tag['TAG'] 
    stops_all['Count_Stop_TGA_TAGrec'] = stops_tag['TGA'] 
    stops_all['Count_Stop_TAA_TAGrec'] = stops_tag['TAA'] 
    
    stops_all['Count_Stop_TAG_TGArec'] = stops_tga['TAG'] 
    stops_all['Count_Stop_TGA_TGArec'] = stops_tga['TGA'] 
    stops_all['Count_Stop_TAA_TGArec'] = stops_tga['TAA'] 

    stops_all['Count_Stop_TAG_TAArec'] = stops_taa['TAG'] 
    stops_all['Count_Stop_TGA_TAArec'] = stops_taa['TGA'] 
    stops_all['Count_Stop_TAA_TAArec'] = stops_taa['TAA'] 


    stops_all = stops_all.drop(['TAG', 'TGA', 'TAA'], axis=1)

    return pd.DataFrame(stops_all)








# def get_all_stops(seqiorec: SeqRecord) -> pd.DataFrame:
#     """
#     Get distribution of STOP codons in a sequence
#     :param seqiorec: SeqRecord object
#     :param window: window size
#     :param step: step size
#     :return:
#     """

#     locations = []
#     stops = []
#     strands = []
#     lengths = []

#     for record in parse_genbank(seqiorec):
#         # Loop over the features
#         for feature in record.features:
#             if feature.type == "CDS":
#                 gene_sequence = feature.extract(record.seq)
#                 stop_codon = gene_sequence[-3:]
#                 #print("Stop codon is %s" % stop_codon)
#                 strand = feature.strand 
#                 if strand == 1: # forward strand
#                     location = feature.location.end - 3
#                     length = abs(feature.location.end - feature.location.start)
#                 else:
#                     location = feature.location.start +3
#                 #print("location is %s" % location)
#                 if stop_codon == "TAA":              
#                     stops.append('TAA')
#                 if stop_codon == "TGA":
#                     stops.append('TGA')
#                 if stop_codon == "TAG":
#                     stops.append('TAG')
#                 locations.append(location)
#                 strands.append(strand)
#                 lengths.append(feature.len)

#     stops_df = pd.DataFrame(
#     {'locations': locations,
#      'stops': stops, 
#      'strands': strands
#     })
    
#     return pd.DataFrame(stops_df)



def get_all_stops(seqiorec: SeqRecord) -> pd.DataFrame:
    """
    Get distribution of STOP codons in a sequence
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """

    locations = []
    stops = []
    strands = []
    lengths = []
    tag_counts = []
    tga_counts = []
    taa_counts = []

    for record in parse_genbank(seqiorec):
        # Loop over the features
        for feature in record.features:
            if feature.type == "CDS":
                length = abs(feature.location.end - feature.location.start)
                gene_sequence = feature.extract(record.seq)
                tag_count = 0
                tga_count = 0
                taa_count = 0

                # counts tags
                for i in range(0, length-2, 3):
                    if gene_sequence[i:i+3] == 'TAG':
                        tag_count += 1
                    if gene_sequence[i:i+3] == 'TGA':
                        tga_count += 1
                    if gene_sequence[i:i+3] == 'TAA':
                        taa_count += 1
                        


                stop_codon = gene_sequence[-3:]
                #print("Stop codon is %s" % stop_codon)
                strand = feature.strand 
                if strand == 1: # forward strand
                    location = feature.location.end - 3
                    length = abs(feature.location.end - feature.location.start)
                else:
                    location = feature.location.start + 3
                #print("location is %s" % location)
                if stop_codon == "TAA":              
                    stops.append('TAA')
                elif stop_codon == "TGA":
                    stops.append('TGA')
                elif stop_codon == "TAG":
                    stops.append('TAG')
                else:
                    stops.append('NA')

                
                strands.append(strand)
                locations.append(location)
                lengths.append(length)
                tag_counts.append(tag_count)
                tga_counts.append(tga_count)
                taa_counts.append(taa_count)

    print(len(locations))
    print(len(stops))
    print(len(strands))
    print(len(lengths))
    print(len(tag_counts))

    stops_df = pd.DataFrame(
    {'locations': locations,
     'stops': stops, 
     'length': lengths,
     'tag_counts': tag_counts,
     'tga_counts': tga_counts,
     'taa_counts': taa_counts
    })
    
    return pd.DataFrame(stops_df)

