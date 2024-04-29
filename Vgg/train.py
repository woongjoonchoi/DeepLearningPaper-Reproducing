# from config import *
# from model import * 
# from dataset_class import *

init_from = 'resume'
import argparse

parser = argparse.ArgumentParser(description='Reproducing VGG network ')

parser.add_argument('--model_version', metavar='--v', type=str, nargs='+',
                    help='moel config version [A,A_lrn,B,C,D,E]')


parser.add_argument('--eval_multi_scale', metavar='eval_scale', type=bool, nargs='+',
                    help='Choose eval multi scale' , required = False)
# print(config.batch_size)
# print(batch_size)
# print(weight_decay)
# print(locals())



args = parser.parse_args()

## argument parsing and configuration
model_version = args.model_version
eval_multi_scale = args.eval_multi_scale

# model = Model_vgg('model_version',num_classes)
if init_from =='scratch' :
    print('scratch  ')
elif init_from =='resume' :
    print('resume')
    
else :
    
    raise Exception("you must set init_from to scratch or resume")
    

# print(args.
# 
# )
