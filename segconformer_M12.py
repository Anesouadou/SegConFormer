import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.functional import relu

import torchvision 
from torchvision import transforms, models
from torchvision.io import read_image 

import torch.optim as optim
import copy

from packages import stemming as st
from packages import AtrousSpatialPyramidPooling as aspp
from packages import ConvBlockAttModule as cbam
from packages import trblock as trb
from packages import UpsampleModule as upsample
from packages import fusion
from packages import decoder
from packages import dataset

class segconformer_M12(nn.Module):
    def __init__(self,n_class,stemming_channels,aspp_opt,tr_channels,fusion_opt,decoder_channels,method='transpose'):
        super().__init__()
        
        in_cha   = stemming_channels[0]
        out_cha1 = stemming_channels[1]
        out_cha2 = stemming_channels[2]
        
        br_cha1 = tr_channels[0]
        br_cha2 = tr_channels[1]
        br_cha3 = tr_channels[2]
        br_cha4 = tr_channels[3]
        
        dec_cha1 = decoder_channels[0]
        dec_cha2 = decoder_channels[1]
        dec_cha3 = decoder_channels[2]
        
        # Stemming
        
        self.two_layer_net = st.ThreeLayerConvNet(in_channels=in_cha,
                                                  out_channels1=out_cha1,
                                                  out_channels2=out_cha2)
        
        # ASPP
        self.aspp_block = aspp.asppblock(aspp_opt)
        
        # CBAM
        self.cbam11  = cbam.CBAM(in_channels = br_cha1,reduction_ratio=8,kernel_size=3)
        self.cbam12  = cbam.CBAM(in_channels = br_cha1,reduction_ratio=8,kernel_size=3)
        self.cbam13  = cbam.CBAM(in_channels = br_cha1,reduction_ratio=8,kernel_size=3)
        self.cbam14  = cbam.CBAM(in_channels = br_cha1,reduction_ratio=8,kernel_size=3)
                                  
        self.cbam21  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)
        self.cbam22  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)
        self.cbam23  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)
        self.cbam24  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)
        self.cbam25  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)
        self.cbam26  = cbam.CBAM(in_channels = br_cha2,reduction_ratio=8,kernel_size=3)

        self.cbamup21  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
                          
        self.cbam31   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam32   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam33   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam34   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam35   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam36   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam37   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam38   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam39   = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam310  = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam311  = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)
        self.cbam312  = cbam.CBAM(in_channels = br_cha3,reduction_ratio=8,kernel_size=3)

        self.cbamup31  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbamup32  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        
        self.cbam41  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbam42  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbam43  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbam44  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbam45  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbam46  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)

        self.cbamup41  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbamup42  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)
        self.cbamup43  = cbam.CBAM(in_channels = br_cha4,reduction_ratio=8,kernel_size=3)

        
        # Tr block
        self.trblock11  = trb.DepthwiseSeparableConv(br_cha1, br_cha1, kernel_size=3)
        self.trblock12  = trb.DepthwiseSeparableConv(br_cha1, br_cha1, kernel_size=3)
        self.trblock13  = trb.DepthwiseSeparableConv(br_cha1, br_cha1, kernel_size=3)
        self.trblock14  = trb.DepthwiseSeparableConv(br_cha1, br_cha4, kernel_size=3)
        
        self.trblock21   = trb.DepthwiseSeparableConv(br_cha2, br_cha2, kernel_size=3)
        self.trblock22   = trb.DepthwiseSeparableConv(br_cha2, br_cha2, kernel_size=3)
        self.trblock23   = trb.DepthwiseSeparableConv(br_cha2, br_cha2, kernel_size=3)
        self.trblock24   = trb.DepthwiseSeparableConv(br_cha2, br_cha2, kernel_size=3)
        self.trblock25   = trb.DepthwiseSeparableConv(br_cha2, br_cha2, kernel_size=3)
        self.trblock26   = trb.DepthwiseSeparableConv(br_cha2, br_cha4, kernel_size=3)

        self.trblockup21 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        
        self.trblock31   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock32   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock33   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock34   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock35   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock36   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock37   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock38   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock39   = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock310  = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock311  = trb.DepthwiseSeparableConv(br_cha3, br_cha3, kernel_size=3)
        self.trblock312  = trb.DepthwiseSeparableConv(br_cha3, br_cha4, kernel_size=3)

        self.trblockup31 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblockup32 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        
        self.trblock41   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblock42   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblock43   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblock44   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblock45   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblock46   = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)

        self.trblockup41 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblockup42 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        self.trblockup43 = trb.DepthwiseSeparableConv(br_cha4, br_cha4, kernel_size=3)
        
        # Upsampling
        # 128 --> 256
        self.branchup1 = upsample.UpsampleModule(in_channels=br_cha4,
                                                 out_channels=br_cha4,
                                                 kernel_size=2,
                                                 method=method,
                                                 scale_factor=2,
                                                 padding=0,
                                                 output_padding=0)

        # 64 -- > 256
        self.branchup21=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup22=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        
        # 32 --> 256
        self.branchup31=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup32=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup33=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        
        # 16 --> 256
        self.branchup41=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup42=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup43=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,
                                                scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        self.branchup44=upsample.UpsampleModule(in_channels=br_cha4,
                                                out_channels=br_cha4,
                                                kernel_size=2,
                                                method=method,scale_factor=2,
                                                padding=0,
                                                output_padding=0)
        
        # Fusion
        if fusion_opt == 'cat':
            fusion_in_cha = br_cha4*len(tr_channels)
            self.fusion_out = fusion.CatModule(in_channels=fusion_in_cha,out_channels=br_cha4)
        elif fusion_opt == 'add':
            self.fusion_out = fusion.AddModule(in_channels=br_cha4,out_channels=br_cha4)
        elif fusion_opt == 'prod':
            self.fusion_out = fusion.ProdModule(in_channels=br_cha4,out_channels=br_cha4)
        
        
        # Decoder
        if dec_cha1 != br_cha4:
            dec_cha1 = copy.copy(br_cha4)
            
        self.decoderblock = decoder.decoderModule(dec_cha1,dec_cha2,dec_cha3)
        
        # Output layer
        self.outconv = nn.Conv2d(dec_cha3, n_class, kernel_size=1)

    def forward(self, x):
        # stemming
        stemblock = self.two_layer_net(x)
        
        # aspp
        branches  = self.aspp_block(stemblock)
        
        # Tr cain of blocks with embedding
        embed11 = self.cbam11(branches[0])
        trb11   = self.trblock11(embed11)
        embed12 = self.cbam12(trb11)
        trb12   = self.trblock12(embed12)
        embed13 = self.cbam13(trb12)
        trb13   = self.trblock13(embed13)
        embed14 = self.cbam14(trb13)
        trb14   = self.trblock14(embed14)
        
        embed21 = self.cbam21(branches[1])
        trb21   = self.trblock21(embed21)
        embed22 = self.cbam22(trb21)
        trb22   = self.trblock22(embed22)
        embed23 = self.cbam23(trb22)
        trb23   = self.trblock23(embed23)
        embed24 = self.cbam24(trb23)
        trb24   = self.trblock24(embed24)
        embed25 = self.cbam25(trb24)
        trb25   = self.trblock25(embed25)
        embed26 = self.cbam26(trb25)
        trb26   = self.trblock26(embed26)
        
        embed31  = self.cbam31(branches[2])
        trb31    = self.trblock31(embed31)
        embed32  = self.cbam32(trb31)
        trb32    = self.trblock32(embed32)
        embed33  = self.cbam33(trb32)
        trb33    = self.trblock33(embed33)
        embed34  = self.cbam34(trb33)
        trb34    = self.trblock34(embed34)
        embed35  = self.cbam35(trb34)
        trb35    = self.trblock35(embed35)
        embed36  = self.cbam36(trb35)
        trb36    = self.trblock36(embed36)
        embed37  = self.cbam37(trb36)
        trb37    = self.trblock37(embed37)
        embed38  = self.cbam38(trb37)
        trb38    = self.trblock38(embed38)
        embed39  = self.cbam39(trb38)
        trb39    = self.trblock39(embed39)
        embed310 = self.cbam310(trb39)
        trb310   = self.trblock310(embed310)
        embed311 = self.cbam311(trb310)
        trb311   = self.trblock311(embed311)
        embed312 = self.cbam312(trb311)
        trb312   = self.trblock312(embed312)
        
        embed41 = self.cbam41(branches[3])
        trb41   = self.trblock41(embed41)
        embed42 = self.cbam42(trb41)
        trb42   = self.trblock42(embed42)
        embed43 = self.cbam43(trb42)
        trb43   = self.trblock43(embed43)
        embed44 = self.cbam44(trb43)
        trb44   = self.trblock44(embed44)
        embed45 = self.cbam45(trb44)
        trb45   = self.trblock45(embed45)
        embed46 = self.cbam46(trb45)
        trb46   = self.trblock46(embed46)
        
        # Upsampling
        ## Br1
        branch1 = self.branchup1(trb14)

        ## Br2
        brup21    = self.branchup21(trb26)
        
        embedup21 = self.cbamup21(brup21)
        trbup21   = self.trblockup21(embedup21)
        
        branch2   = self.branchup22(trbup21)

        ## Br3
        brup31    = self.branchup31(trb312)
        
        embedup31 = self.cbamup31(brup31)
        trbup31   = self.trblockup31(embedup31)
        
        brup32    = self.branchup32(trbup31)
        
        embedup32 = self.cbamup32(brup32)
        trbup32   = self.trblockup32(embedup32)
        
        branch3   = self.branchup33(trbup32)

        ## Br4
        brup41 = self.branchup41(trb46)

        embedup41 = self.cbamup41(brup41)
        trbup41   = self.trblockup41(embedup41)
        
        brup42    = self.branchup42(trbup41)
        
        embedup42 = self.cbamup42(brup42)
        trbup42   = self.trblockup42(embedup42)
        
        brup43    = self.branchup43(trbup42)
        
        embedup43 = self.cbamup43(brup43)
        trbup43   = self.trblockup43(embedup43)
        
        branch4   = self.branchup44(trbup43)

        # Fusion
        tensor_seq  = (branch1,branch2,branch3,branch4)
        fusion_cat  = self.fusion_out(tensor_seq)
        
        # Decoder
        decoder_out = self.decoderblock(fusion_cat)

        # Output layer
        output = self.outconv(decoder_out)           
        
        return output
        
    #He initialization for the weights
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)        
        