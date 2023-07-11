import torch
import torch.nn as nn
from data import transforms as T
import math

from torch.nn import functional as F
class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self, x, k0, mask, sensitivity,coilmask):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        coilmask = coilmask.unsqueeze(4)
        #print("gamma_input: ", gamma_input.shape)
        sensitivity_new = torch.mul(sensitivity,coilmask)
        #print("gamma_input: ", gamma_input.shape,"sens: ",sensitivity.shape,"sens new: ",sensitivity_new.shape)
        #print(torch.sum(sensitivity_new[0,0,:,:,:]), torch.sum(sensitivity_new[0,1,:,:,:]),torch.sum(sensitivity_new[0,3,:,:,:]),torch.sum(sensitivity_new[0,5,:,:,:]))
        x = T.complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), 
                               sensitivity_new[...,0], sensitivity_new[...,1])
     
        #k = torch.fft(x, 2, normalized=True)
        x = x[...,0] + x[...,1]*1j
        k = torch.fft.fft2(x,norm='ortho')
        k = torch.stack([k.real, k.imag], dim=-1)
              
        v = self.noise_lvl
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        #x = torch.ifft(out, 2, normalized=True)
        out = out[...,0] + out[...,1]*1j
        x = torch.fft.ifft2(out, norm='ortho')
        x = torch.stack([x.real, x.imag], dim=-1)
 
        Sx = T.complex_multiply(x[...,0], x[...,1], 
                                sensitivity_new[...,0], 
                               -sensitivity_new[...,1]).sum(dim=1)     
          
        SS = T.complex_multiply(sensitivity[...,0], 
                                sensitivity[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)
        
        return Sx, SS

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )
        
                  
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        out = self.layers(input)
        
        return out

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class UnetModelWithPyramidDWPAndDC(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, contextvectorsize):
        super().__init__()
        self.relu = nn.ReLU() 

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.weightsize = []

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.dwpfilterbank = nn.ModuleList([nn.Sequential(
            nn.Linear(contextvectorsize, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8,chans*chans*3*3),
        )])

        #self.weightsize.append([chans,in_chans,3,3])
        self.weightsize.append([chans,chans,3,3])
        #print("self.weightsize: ",self.weightsize)

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            self.dwpfilterbank += [nn.Sequential(nn.Linear(contextvectorsize,8), nn.ReLU(), nn.Linear(8,8), nn.ReLU(), nn.Linear(8,(ch*2)*(ch*2)*3*3))]
            #self.weightsize.append([ch*2,ch,3,3])
            self.weightsize.append([ch*2,ch*2,3,3])
            ch *= 2
            #print("self.weightsize: ",self.weightsize)
##################bottleneck layers #########################3
        #self.conv = ConvBlock(ch, ch, drop_prob)
        self.dwp_latent = nn.Sequential(
            nn.Linear(contextvectorsize, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8,ch*ch*3*3),
        )
        self.latentweightsize = [ch,ch,3,3]
        #print("self.latentweightsize: ", self.latentweightsize)
        self.latentinstancenorm=nn.InstanceNorm2d(ch,affine=True)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            #print("ch: ", ch)
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        #self.dc = DataConsistencyLayer(args.usmask_path, args.device)

    def forward(self,x, gamma_val):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        batch_size = x.size(0)
        batch_outputs=[]
        for n in range(batch_size):
            stack = []
            output = x[n]
            output = output.unsqueeze(0)
            #print("input size: ", output.size())
            xtemp = output
            filterbank=[]
        # Apply down-sampling layers
            for layer in self.down_sample_layers:
                output = layer(output)
                #print("downsample output size: ", output.size())
                stack.append(output)
                output = F.max_pool2d(output, kernel_size=2)
                #print("downsample output size after maxpool: ", output.size())

            for dwp,wtsize in zip(self.dwpfilterbank,self.weightsize):
                #print("gamma shape: ",gamma_val.shape)
                filters = dwp(gamma_val[n])
                #print("filers size: ", filters.size()," weights size : ",wtsize)
                filters = torch.reshape(filters,wtsize)
                filterbank.append(filters)

        #output = self.conv(output)
            latentfilters = self.dwp_latent(gamma_val[n])
            #print("latent filers size: ", latentfilters.size()," weights size : ",self.latentweightsize)
            latentweights = torch.reshape(latentfilters, self.latentweightsize)
            output = self.relu(self.latentinstancenorm(F.conv2d(output, latentweights, bias=None, stride=1, padding=1)))
            output_latent = output
            #print("latent output size: ", output.size())
        # Apply up-sampling layers
            for layer in self.up_sample_layers:
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                #print("upsample output size: ", output.size())
                encoutput = stack.pop()
                #print("encoutput size: ", encoutput.size())
                encoutfinal = F.conv2d(encoutput,filterbank.pop(),bias=None,stride=1,padding=1)
                output = torch.cat([output, encoutfinal], dim=1)
                #print("output size after cat: ", output.size())
                output = layer(output)
            output = self.conv2(output)
            #output=output+xtemp
            #print("output shape: ",output.shape," k[n] shape: ",k[n].shape," mask[n] shape: ",mask[n].shape) 
            #output = self.dc(output,k[n],acc_string[n], mask_string[n], dataset_string[n],mask[n])
            batch_outputs.append(output)
        output = torch.cat(batch_outputs,dim=0)
        return output


class networkwithunet(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(networkwithunet, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):

            #unetmodel = UnetModel(in_chans=2,out_chans=2,chans=32,num_pool_layers=3,drop_prob=0.0)
            
            unetmodel = UnetModelWithPyramidDWPAndDC(2,2,32,3,0.0,15)
            conv_blocks.append(unetmodel) 
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
        print(self.conv_blocks)
        print(self.dc_blocks)
        print(self.wa_blocks)

    def pad(self,x):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        #print(w_pad,h_pad)
        # # TODO: fix this type when PyTorch fixes theirs
        # # the documentation lies - this actually takes a list
        # # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self,x,h_pad,w_pad,h_mult,w_mult):
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]


    def forward(self, x, k, m, c, gamma_input, coil_mask):
                
        for i in range(self.cascades):
            x_cnn = x.permute(0, 3, 1, 2)
            x_cnn,padinfo=self.pad(x_cnn)
            x_cnn = self.conv_blocks[i](x_cnn, gamma_input)
            x_cnn = self.unpad(x_cnn,padinfo[0],padinfo[1],padinfo[2],padinfo[3])
            x_cnn = x_cnn.permute(0, 2, 3, 1)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c, coil_mask)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
        return x    
 
    
