DatasetName = 'Cifar' # Cifar ,CalTech

## model configuration

num_classes =   100# CalTech 257 Cifar 100 
model_version = None ## you must configure it. 

## data configuration

train_min = 256
train_max = None
test_min = 256
test_max = 256

## train configuration

batch_size = 64
lr = 1e-2
momentum = 0.9
weight_decay  = 5e-4
lr_factor = 0.1
epoch = 74

train_multi_scale = False ## default, if you want multi scale, overwrite it to True
eval_multi_scale = False 
update_count = int(256/batch_size)
accum_step = int(256/batch_size)
eval_step = 20

## resume



init_from ='scratch' ## if you want load checkpoint, you can use it.