'''search.py
 This file may be used for kicking off the model, reading in data and preprocessing
 '''

import argparse
import sys
from utils import *
from gutenberg_data import *

sys.path.append("./models")
from baseline import *

# Read in command line arguments to the system
def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BASELINE', help="Model to run")
    parser.add_argument('--train_type', type=str, default="GUTENBERG", help="Data type - Gutenberg or custom")
    parser.add_argument('--train_path', type=str, default='data/american/', help='Path to the training set')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    print(args)

    if args.model == 'BASELINE':
        # Get books from train path and call baseline model train function
        if (args.train_type == 'GUTENBERG'):
            train_data =  gutenberg_dataset(args.train_path)
            print("training")
            train_baseline(train_data)
            print("testing")            
            # Implement testing

    elif args.model == 'ENCDEC':
        pass
    elif args.model == 'VAE':
        pass

    else:
        raise Exception("Please select appropriate model")