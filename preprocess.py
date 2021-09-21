import json
import argparse
from datetime import datetime
import calendar
import numpy as np

def parse_args() -> argparse.Namespace:
    """
    Returns:
        arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--maxMZ", type=int, default=20000)
    parser.add_argument("--minMZ", type=int, default=2000)

    args = parser.parse_args()
    return args


def read_file(args):
    filename=args.input
    mapping = {'S': 0, 'R': 1}
    with open(filename) as f:
        lines = f.read().splitlines()
        data_length=len(lines)
        print(data_length)
        #print(lines[7894])        
        
        mv_pos_max=args.maxMZ
        mv_pos_min=args.minMZ

        mv_data= np.zeros((data_length-1,18000),float)
        labels= np.zeros(data_length-1)
        for i in range(len(lines)):
            if i==0:
                continue
            print(i)
            line=lines[i]
            elements = line.strip().split(',')
            intensity=elements[5][1:-1].split(';')
            mv_quantize=elements[7][1:-1].split(';')
    
            #print(elements[5])
            if (len(intensity) != len(mv_quantize)):
                print("dimesion error!!!!!!")
                print(len(intensity))
                print(len(mv_quantize))
                print(elements[0])
                print(elements[1])
    
            mv_vec = np.zeros(18000)
            for j in range(len(mv_quantize)):
                mv_pos_int=int(mv_quantize[j])
                intensity_value=int(intensity[j])
                if mv_pos_int > mv_pos_max:
                    print("the value of m/z over 20000: {}".format(mv_pos_int))
                    mv_pos_int =mv_pos_max
                if mv_pos_int < mv_pos_min:
                    print("the value of m/z lower 2000: {}".format(mv_pos_int))
                    mv_pos_int =mv_pos_min
                if mv_vec[mv_pos_int-2000]>0:
                    print("duplicate!!!!")
                    print(mv_pos_int)
                mv_vec[mv_pos_int-2000]=intensity_value
            #print(mv_vec)
            mv_data[i-1]=mv_vec
            labels[i-1]=mapping[elements[6].replace("\"","")]
            #mv_data = np.append(mv_data, [mv_vec], axis=0)
    
            #print(elements[4])
    return mv_data,labels



def main():
    args=parse_args()

    mv_data,labels = read_file(args)
    # /Users/tsung-ting/Documents/lab3_volume/MALDI-TOF/MALDI-TOF/20210414/Linkou_EF_Data (round off).csv
    elements=args.input.split('/')[-1].split('.')
    print(elements[0])
    np.save("{}_mz_dim".format(elements[0]), mv_data)
    np.savetxt("{}_labels.csv".format(elements[0]), labels)

if __name__ == '__main__':
    main()