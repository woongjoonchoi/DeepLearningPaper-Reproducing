DatasetName = 'ImageNet' # Cifar ,CalTech ,Cifar10, Mnist , ImageNet

## model configuration

num_classes =   1000  
# CalTech 257 Cifar 100  Cifar10 10 ,Mnist 10 ImageNet 1000
model_version = None ## you must configure it. 

## data configuration

train_min = 256
train_max = None
test_min = 256
test_max = 256

## train configuration

batch_size = 128
lr = 1e-2
momentum = 0.9
weight_decay  = 5e-4
lr_factor = 0.1
epoch = 4
nestrov = False  # ImageNet D nestrov True
clip= None # model D grad clip 0.7

train_multi_scale = False ## default, if you want multi scale, overwrite it to True
eval_multi_scale = False 
update_count = int(256/batch_size)
accum_step = int(256/batch_size)
eval_step =26 * accum_step  ## CalTech 5 Cifar 5 Mnist 6 , Cifar10 5 ImageNet  26


## model configuration
xavier_count= 4 

last_xavier = -8  ##  

except_xavier = None

model_layers =None


## resume

project_id = '57mvhi3u'

init_from ='resume' ## if you want load checkpoint, you can use it. resume or scratch