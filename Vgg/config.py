DatasetName = 'CalTech'

## model configuration
batch_size = 128
lr = 1e-2
momentum = 0.9
weight_decay  = 5e-4
lr_factor = 0.1
num_classes = 1000
model_version = None ## you must configure it. 

## data configuration

s_min = 256
s_max = None
train_multi_scale = False ## default, if you want multi scale, overwrite it to True
eval_multi_scale = False 

## resume

init_from ='scratch' ## if you want load checkpoint, you can use it.