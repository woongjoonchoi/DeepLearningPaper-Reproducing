from config import *
from model import * 
from dataset_class import *

import argparse

import torch.optim as optim
import pickle
from tqdm import tqdm 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device :{device}')
# init_from = 'resume'

parser = argparse.ArgumentParser(description='Reproducing VGG network ')

parser.add_argument('--model_version', metavar='--v', type=str, nargs='+',
                    help='moel config version [A,A_lrn,B,C,D,E]')


parser.add_argument('--eval_multi_scale', metavar='eval_scale', type=bool, nargs='+',
                    help='Choose eval multi scale' , required = False)



args = parser.parse_args()


## top - k error caculate

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # print(f'top {k}')
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res

## chkpoint to current cpu

## argument parsing and configuration
model_laeyrs_num={"A" : 11,"B":13,"C":16,"D":16,"E":19}

model_version = args.model_version[0]
eval_multi_scale = args.eval_multi_scale

model_layers=model_laeyrs_num[model_version]


## wandb login and project setting
import wandb
wandb.login()   
if init_from =='resume'  :
    wandb.init(
        # Set the project where this run will be logged
        project="vgg-2",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        id = project_id,
        resume="must"
    )

else :
    wandb.init(
        # Set the project where this run will be logged
        project="vgg-2",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"vgg_{model_version}_dataset_{DatasetName}trainmin_{train_min}_testmin{test_min}",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": f"vgg_{model_version}_trainmin_{train_min}_testmin{test_min}",
        "dataset": DatasetName,
        "epochs": 74,
        "batch size" : batch_size
        })





## neuralnet and optimizer configuration
model = Model_vgg(model_version,num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,nesterov=nestrov,momentum=momentum)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10,threshold=1e-3,eps = 1e-5)

if nestrov :
    print('nestrov sgd momentum')
    print(optimizer)
# if init_from =='scratch' :
#     print('Training initializie from scratch  ')
# elif init_from =='resume' :
#     print('resume')
#     load_pt = torch.load()
    
# else :
    
#     raise Exception("you must set init_from to scratch or resume")
    
    
## dataset
# all_dataset=None
# target_list = None
if DatasetName == 'CalTech' :
    real_caltech = Custom_Caltech(root=os.getcwd(),s_min=train_min)
    with open("target_list.pickle" , 'rb') as f:
        target_list = pickle.load(f)
    X_train,X_test  = train_test_split(list(range(len(real_caltech)))  ,test_size=0.2,random_state= 42,stratify=target_list)
    train_data ,val_data = Subset(real_caltech,X_train) ,Subset(real_caltech,X_test)
    val_data.dataset.val= True
    val_data.dataset.s_min = test_min
    val_data.dataset.transform=    A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=val_data.dataset.S),
                        A.CenterCrop(height =224,width=224),
                        # A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

                )

elif DatasetName == 'Cifar' :
    train_data = Custom_Cifar(root=os.getcwd(),download=True)
    val_data  = Custom_Cifar(root=os.getcwd(),train=False,download=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose(
                    [
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224),
                        # A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

                )
    
elif DatasetName == 'Cifar10' :
    train_data = Custom_Cifar_10(root=os.getcwd(),download=True)
    val_data  = Custom_Cifar_10(root=os.getcwd(),train=False,download=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose(
                    [
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224),
                        # A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

                )
    
elif DatasetName == 'Mnist' :
    train_data = Cusotm_MNIST(root=os.getcwd(),download=True)
    val_data  = Cusotm_MNIST(root=os.getcwd(),train=False,download=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224),
                        # A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

                )

elif DatasetName == 'ImageNet' :
    train_data= Cusotm_ImageNet(root='ImageNet',split='train')
    val_data= Cusotm_ImageNet('ImageNet',split='val',val=True)
    val_data.val= True
    val_data.s_min = test_min
    val_data.transform=    A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=val_data.S),
                        A.CenterCrop(height =224,width=224),
                        # A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

                )
    

train_loader = torch.utils.data.DataLoader(train_data,batch_size= batch_size,shuffle = True , num_workers=4,pin_memory = True,prefetch_factor = 2,drop_last = True)
val_loader = torch.utils.data.DataLoader(val_data,batch_size= batch_size,shuffle = True , num_workers=4,pin_memory = True,prefetch_factor = 2,drop_last = True)
  


best_val_loss=None
if except_xavier is None :
    save_checkpoint_name = f"vgg_{model_version}_{batch_size}_trainmin_{train_min}_testmin{test_min}_dataset_{DatasetName}_{xavier_count}_{model_layers+last_xavier-1}.pt"
else :
    save_checkpoint_name = f"vgg_{model_version}_{batch_size}_trainmin_{train_min}_testmin{test_min}_dataset_{DatasetName}_{xavier_count}_{model_layers+last_xavier-1}_{except_xavier}.pt"
resume_epoch= 1
if init_from =='scratch' :
    print('Training initializie from scratch  ')
elif init_from =='resume' :
    print('resume')
    load_pt = torch.load(save_checkpoint_name)
    model.load_state_dict(load_pt['model_state_dict'])
    model.to(device)  ## model을 cuda tensor로 부터 resume할때 이코드가 필요하다.
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10,threshold=1e-3,eps = 1e-5)
    optimizer.load_state_dict(load_pt['optimizer_state_dict'])
    optimizer.zero_grad(set_to_none=True)
    resume_epoch = load_pt['epoch'] 
    print(f"resume from epoch  :  {resume_epoch }  ")
    print(f"resume from optimizer ")
    print(optimizer)
    
    
else :
    
    raise Exception("you must set init_from to scratch or resume")

model = model.to(device)


print(save_checkpoint_name)
grad_clip = 1.0

print(f'grad clip : {grad_clip}')
print(model)

wandb.watch(model,log="all", log_freq=26)
for e in range(epoch-resume_epoch) :
    print(f'Training Epoch : {e+resume_epoch}')
    # if e == 0  and init_from == 'resume':
    #     e =load_pt['epoch']
    #     print()
    total_loss = 0 
    val_iter = iter(val_loader)
    train_acc=[0,0]
    train_num = 0
    
    total_acc = [0,0]
    count= 0
    for i , data in tqdm(enumerate(train_loader)) :
        
        
        model.train()
        # model.zero_grad(set_to_none=True)  ## optimizer로만 gradient를 초기화 해주자. 
        img,label= data
        img,label =img.to(device, non_blocking=True) ,label.to(device, non_blocking=True)
        
        output = model(img)
        
        loss = criterion(output,label) /accum_step
        
        temp_output ,temp_label = output.detach().to('cpu') , label.detach().to('cpu')
        temp_acc = accuracy(temp_output,temp_label,(1,5))
        train_acc=[train_acc[0]+temp_acc[0] , train_acc[1]+temp_acc[1]]
        train_num+=batch_size
        temp_output,temp_label,temp_acc = None,None,None
        
        loss.backward()
        total_loss += loss.detach().to('cpu')
        img,label=None,None
        torch.cuda.empty_cache()
        if i> 0 and i%update_count == 0 :
            print(f'Training steps : {i}  parameter update loss :{total_loss} ')
            if grad_clip is not None:
                print(f'Training steps : {i}  parameter grad clip to {grad_clip}')
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if total_loss < 7.0 :
                print(f"train loss {total_loss}less than 7.0  ,set grad clip to {clip}")
                grad_clip = clip 
            if i % eval_step != 0 :  
                total_loss = 0 
                
            output,loss = None,None
            torch.cuda.empty_cache()
        if i>0 and i % eval_step == 0 :
            
            print(f'train losss :{total_loss}')
            wandb.log({'train/loss' : total_loss})
            temp_loss = total_loss
            total_loss= 0
             
            val_loss = 0
            torch.cuda.empty_cache()

            for j   in tqdm(range(update_count)) :
                loss = None
                print(f'Evaluation Steps Start')
                try :
                    img,label = next(val_iter)
                except StopIteration :
                    val_iter= iter(val_loader)
                    img,label = next(val_iter)
                with torch.no_grad():
                    model.eval()
                    
                    img , label = img.to(device, non_blocking=True) , label.to(device, non_blocking=True)
                    output = model(img)
                    temp_output ,temp_label = output.detach().to('cpu') , label.detach().to('cpu')
                    temp_acc = accuracy(temp_output,temp_label,(1,5))
                    total_acc=[total_acc[0]+temp_acc[0] , total_acc[1]+temp_acc[1]]
                    count+=batch_size
                    
                    loss = criterion(output,label)/accum_step
                    val_loss += loss.detach().to('cpu') 
                    # loss.backward()
                    torch.cuda.empty_cache()
            
   
                    img,label,output ,loss= None,None,None,None

 

                torch.cuda.empty_cache()



            wandb.log({'val/loss' : val_loss})
            if abs(val_loss-temp_loss) > 0.03 :
                grad_clip=clip
                print(f"val_loss {val_loss} - train_loss {temp_loss} = {abs(val_loss-temp_loss)} > 0.3")
                print(f"set grad clip to {grad_clip}")
            if best_val_loss is None or best_val_loss - val_loss >1e-4 :
                print(f'torch best model save steps {i} best_loss {val_loss} ')
                torch.save(
                        {
                            'epoch' :  e+resume_epoch ,
                            'model_state_dict' : model.state_dict() , 
                            'optimizer_state_dict' : optimizer.state_dict(),
                            'loss' : val_loss,
                            'steps'  : i
                        } , save_checkpoint_name
                    )
                best_val_loss = val_loss
                
            val_loss = None
        # optimizer.zero_grad()
        img,label,output = None,None,None
    
    # total_acc = [0,0]
    # count= 0
    # for i , data in enumerate(val_loader) :
    #     count +=  batch_size
    #     with torch.no_grad() :
    #         model.eval()
    #         img,label = data
    #         img , label = img.to(device, non_blocking=True) , label.to(device, non_blocking=True)
    #         output = model(img)
    #         acc = accuracy(output.detach().to('cpu'),label.detach().to('cpu'),(1,5))
    #         total_acc=[total_acc[0]+acc[0] , total_acc[1]+acc[1]]
    
    print(f'top 1 val acc : {total_acc[0]}  top 5 val acc : {total_acc[1]}')
    print(f'val_size :{count}')
    top_1_acc ,top_5_acc   = 100*total_acc[0]/count, 100*total_acc[1]/count
    print(f'top 1 val acc  %: {top_1_acc}')
    print(f'top 5 val acc  %: {top_5_acc}')
    wandb.log({'val/top-1-error' : 100-top_1_acc})
    wandb.log({'val/top-5-error' : 100-top_5_acc})
    
    print(f'top 1 train acc : {train_acc[0]}  top 5 train acc : {train_acc[1]}')
    print(f'train_size :{train_num}')
    top_1_train ,top_5_train   = 100*train_acc[0]/train_num, 100*train_acc[1]/train_num
    print(f'top 1 train acc  %: {top_1_train}')
    print(f'top 5 train acc  %: {top_5_train}')
    wandb.log({'train/top-1-error' : 100-top_1_train})
    wandb.log({'train/top-5-error' : 100-top_5_train})
    
    scheduler.step(top_5_acc)
    wandb.log({'lr' : optimizer.param_groups[0]['lr']})
    wandb.log({'epoch' : e +resume_epoch})
    
    
chkpoint = torch.load(save_checkpoint_name)
# model
chkpoint['epoch']+=1

torch.save(chkpoint,save_checkpoint_name)
    
wandb.alert(title= "Finished Training " ,text = "Finish Training ")