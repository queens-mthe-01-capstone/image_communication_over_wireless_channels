
from train.cosq_train_helper_fast import generate_and_save_codebooks
import sys

def main():
    
    # Redirect print statements to a log file
    log_file = open('cosq_log.txt', 'w')
    sys.stdout = log_file

    
    rates = [3, 4, 5, 6, 7, 8]
    codebook_types = ['dc', 'ac']

    for codebook_type in codebook_types:
        generate_and_save_codebooks(rates, codebook_type)

    # Close the log file
    log_file.close()

if __name__ == "__main__":
    main()