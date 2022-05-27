Copies with files for FID calculation. 

Scripts assume that ```./generated/``` exists in the current directory and that it contains generated images

Each FID result is written to the current directory to a separate CSV file

all copies are meant to be distributed across 3 GPU instances

cuda:0 is specified in scripts 0-1

cuda:1 is specified in scripts 2-4

cuda:2 is specified in scripts 5-7
