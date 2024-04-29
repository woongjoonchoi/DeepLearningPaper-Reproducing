import torch
from torch import nn

Config_channels = {
"A" : [64,"M" , 128,  "M"  , 256,256,"M" ,512,512 ,"M" , 512,512,"M"] ,
"A_lrn" : [64,"LRN","M" , 128,  "M"  , 256,256,"M" ,512,512 ,"M" , 512,512,"M"] ,
"B" :[64,64,"M" , 128,128,  "M"  , 256,256,"M" ,512,512 ,"M" , 512,512,"M"]  ,
"C" : [64,64,"M" , 128,128,  "M"  , 256,256,256,"M" ,512,512 ,512,"M" , 512,512,512,"M"] ,
"D" :[64,64,"M" , 128,128,  "M"  , 256,256,256,"M" ,512,512 ,512,"M" , 512,512,512,"M"] ,
"E" :[64,64,"M" , 128,128,  "M"  , 256,256,256,256,"M" ,512,512 ,512,512,"M" , 512,512,512,512,"M"]         ,

}



Config_kernel = {
"A" : [3,2 , 3,  2  , 3,3,2 ,3,3 ,2 , 3,3,2] ,
"A_lrn" : [3,2,2 , 3,  2  , 3,3,2 ,3,3 ,2 , 3,3,2] ,
"B" :[3,3,2 , 3,3,  2  , 3,3,2 ,3,3 ,2 , 3,3,2]  ,
"C" : [3,3,2 , 3,3,  2  , 3,3,1,2 ,3,3 ,1,2 , 3,3,1,2] ,
"D" :[3,3,2 , 3,3,  2  , 3,3,3,2 ,3,3 ,3,2 , 3,3,3,2] ,
"E" :[3,3,2 , 3,3,  2  , 3,3,3,3,2 ,3,3 ,3,3,2 , 3,3,3,3,2]         ,

}

# def make_feature_extractor(cfg_c,cfg_k):
#     feature_extract = []
#     in_channels = 3
#     i = 1
#     for  out_channels , kernel in zip(cfg_c,cfg_k) :
#         # print(f"{i} th layer {out_channels} processing")
#         if out_channels == "M" :
#             feature_extract += [nn.MaxPool2d(kernel,2) ]
#         elif out_channels == "LRN":
#             feature_extract += [nn.LocalResponseNorm(5,k=2) , nn.ReLU()]
#         elif out_channels == 1:
#             feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1) , nn.ReLU()]
#         else :
#             feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1 , padding = 1) , nn.ReLU()]

#         if isinstance(out_channels,int) :   in_channels = out_channels
#         i+=1
#     return nn.Sequential(*feature_extract)


class Model_vgg(nn.Module) :
    def __init__(self,version , num_classes):
        conv_5_out_w ,conv_5_out_h = 7,7
        conv_5_out_dim =512
        conv_1_by_1_1_outchannel = 4096
        conv_1_by_1_2_outchannel = 4096
        # conv_1_by_1_3_outchannel = num_classes
        super().__init__()
        self.feature_extractor = self.make_feature_extractor(Config_channels[version] , Config_kernel[version])

        self.output_layer = nn.Sequential(
                             nn.Conv2d(conv_5_out_dim  ,conv_1_by_1_1_outchannel ,7) ,
                             nn.ReLU(),
                             nn.Dropout2d(),
                             nn.Conv2d(conv_1_by_1_1_outchannel ,conv_1_by_1_2_outchannel,1 ) ,
                             nn.ReLU(),
                             nn.Dropout2d(),
                             nn.Conv2d(conv_1_by_1_2_outchannel ,num_classes,1 )
                             )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.apply(self._init_weights)
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        x= self.avgpool(x)
        x= torch.flatten(x,start_dim = 1)
        return x


    @torch.no_grad()
    def _init_weights(self,m):
        if isinstance(m,nn.Conv2d):
            # print(m)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None :
                nn.init.zeros_(m.bias)

    #     pass
    def make_feature_extractor(self,cfg_c,cfg_k):
        feature_extract = []
        in_channels = 3
        i = 1
        for  out_channels , kernel in zip(cfg_c,cfg_k) :
            # print(f"{i} th layer {out_channels} processing")
            if out_channels == "M" :
                feature_extract += [nn.MaxPool2d(kernel,2) ]
            elif out_channels == "LRN":
                feature_extract += [nn.LocalResponseNorm(5,k=2) , nn.ReLU()]
            elif out_channels == 1:
                feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1) , nn.ReLU()]
            else :
                feature_extract+= [nn.Conv2d(in_channels,out_channels,kernel,stride = 1 , padding = 1) , nn.ReLU()]

            if isinstance(out_channels,int) :   in_channels = out_channels
            i+=1
        return nn.Sequential(*feature_extract)
