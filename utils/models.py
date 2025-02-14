import torchvision.models as torchmodels
from torch.hub import load_state_dict_from_url
from PIL import Image
from Models.open_ipcl import models as ipclmodels
from torchvision.transforms import Compose, Resize, ToTensor,Normalize
import torch
import Models.SLIP.models as slipmodels
from Models.TDANN.demo.src.model import load_model_from_checkpoint as tdannmodels

# Load models and transforms with weights

def TDANN(type='TDANN',lw=1):
    LW_MODS = [ "", "_lwx2", "_lwx5", "_lwx10", "_lwx100","_lw01"][lw]
    bases = {
        "TDANN": "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
        "Categorization": "supervised_spatial_resnet18_swappedon_SineGrating2019",
        "AbsoluteSL": "simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl",
    }
    
    model = tdannmodels(checkpoint_path='/home/ansh/bigdata/ModelWeights/TDANNcheckpoints/{}{}_checkpoints/model_final_checkpoint_phase199.torch'.format(bases[type],LW_MODS))
    transform=torchmodels.ResNet18_Weights.DEFAULT.transforms()
    
    model.cuda()
    model.eval()


    return model,transform
    

    
    





def ALEXNET():
    model=torchmodels.alexnet(weights='DEFAULT')
    transform=torchmodels.AlexNet_Weights.DEFAULT.transforms()
    
    model.cuda()
    model.eval()
    return model,transform

def RESNET(size='resnet50'):
    #['resnet18','resnet34','resnet50','resnet101','resnet152']
    if size=='resnet18':
        model=torchmodels.resnet18(weights='DEFAULT')
        transform=torchmodels.ResNet18_Weights.DEFAULT.transforms()
    if size=='resnet34':
        model=torchmodels.resnet34(weights='DEFAULT')
        transform=torchmodels.ResNet34_Weights.DEFAULT.transforms()
    if size=='resnet50':
        model=torchmodels.resnet50(weights='DEFAULT')
        transform=torchmodels.ResNet50_Weights.DEFAULT.transforms()
    if size=='resnet50u':
        model=torchmodels.resnet50(weights=None)
        transform=torchmodels.ResNet50_Weights.DEFAULT.transforms()
    if size=='resnet101':
        model=torchmodels.resnet101(weights='DEFAULT')
        transform=torchmodels.ResNet101_Weights.DEFAULT.transforms()
    if size=='resnet152':
        model=torchmodels.resnet152(weights='DEFAULT')
        transform=torchmodels.ResNet152_Weights.DEFAULT.transforms()

    model.cuda()
    model.eval()
    return model,transform


def VGG(size='vgg16bn'):
    #['vgg11','vgg11bn','vgg13','vgg13bn','vgg16','vgg16bn','vgg19','vgg19bn']
    if size=='vgg11':
        model=torchmodels.vgg11(weights='DEFAULT')
        transform=torchmodels.VGG11_Weights.DEFAULT.transforms()
    if size=='vgg11bn':
        model=torchmodels.vgg11_bn(weights='DEFAULT')
        transform=torchmodels.VGG11_BN_Weights.DEFAULT.transforms()
    if size=='vgg13':
        model=torchmodels.vgg13(weights='DEFAULT')
        transform=torchmodels.VGG13_Weights.DEFAULT.transforms()
    if size=='vgg13bn':
        model=torchmodels.vgg13_bn(weights='DEFAULT')
        transform=torchmodels.VGG13_BN_Weights.DEFAULT.transforms()
    if size=='vgg16':
        model=torchmodels.vgg16(weights='DEFAULT')
        transform=torchmodels.VGG16_Weights.DEFAULT.transforms()
    if size=='vgg16bn':
        model=torchmodels.vgg16_bn(weights='DEFAULT')
        transform=torchmodels.VGG16_BN_Weights.DEFAULT.transforms()
    if size=='vgg16bnu':
        model=torchmodels.vgg16_bn(weights=None)
        transform=torchmodels.VGG16_BN_Weights.DEFAULT.transforms()
    if size=='vgg19':
        model=torchmodels.vgg19(weights='DEFAULT')
        transform=torchmodels.VGG19_Weights.DEFAULT.transforms()
    if size=='vgg19bn':
        model=torchmodels.vgg19_bn(weights='DEFAULT')
        transform=torchmodels.VGG19_BN_Weights.DEFAULT.transforms()

    model.cuda()
    model.eval()
    return model,transform




def VIToriginal(size='b16'):
    #['b16','b32','l16','l32','h14']
    if size=='b16':
        model=torchmodels.vit_b_16(weights='DEFAULT')
        transform=torchmodels.ViT_B_16_Weights.DEFAULT.transforms()
    if size=='b32':
        model=torchmodels.vit_b_32(weights='DEFAULT')
        transform=torchmodels.ViT_B_32_Weights.DEFAULT.transforms()
    if size=='b32u':
        model=torchmodels.vit_b_32(weights=None)
        transform=torchmodels.ViT_B_32_Weights.DEFAULT.transforms()
    if size=='l16':
        model=torchmodels.vit_l_16(weights='DEFAULT')
        transform=torchmodels.ViT_L_16_Weights.DEFAULT.transforms()
    if size=='l32':
        model=torchmodels.vit_l_32(weights='DEFAULT')
        transform=torchmodels.ViT_L_32_Weights.DEFAULT.transforms()
    if size=='h14':
        model=torchmodels.vit_h_14(weights='DEFAULT')
        transform=torchmodels.ViT_H_14_Weights.DEFAULT.transforms()
    model.cuda()
    model.eval()
    return model,transform







def SLIPMODELS(type='slip',size='base'):
    transform = Compose([ToTensor(),Resize((224, 224))])
    #['slip','clip','simclr']
    if type=='slip':
        if size=='small':
            model=slipmodels.SLIP_VITS16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='base':
            model=slipmodels.SLIP_VITB16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='large':
            model=slipmodels.SLIP_VITL16(ssl_mlp_dim=4096,ssl_emb_dim=256)
    if type=='clip':
        if size=='small':
            model=slipmodels.CLIP_VITS16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='base':
            model=slipmodels.CLIP_VITB16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='large':
            model=slipmodels.CLIP_VITL16(ssl_mlp_dim=4096,ssl_emb_dim=256)
    if type=='simclr':
        if size=='small':
            model=slipmodels.SIMCLR_VITS16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='base':
            model=slipmodels.SIMCLR_VITB16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='large':
            model=slipmodels.SIMCLR_VITL16(ssl_mlp_dim=4096,ssl_emb_dim=256)
    if type == 'untrained':
        if size=='small':
            model=slipmodels.SLIP_VITS16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='base':
            model=slipmodels.SLIP_VITB16(ssl_mlp_dim=4096,ssl_emb_dim=256)
        if size=='large':
            model=slipmodels.SLIP_VITL16(ssl_mlp_dim=4096,ssl_emb_dim=256) 
    if type != 'untrained':
        state_dict = torch.load('/home/ansh/bigdata/ModelWeights/SlipWeights/'+type+'_'+size+'_25ep.pt')['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    model=model.visual
    model.eval()
    model.cuda()
    return model,transform






def IPCLMODEL(type='ipcl'):
    #ipcl
    if type=='ipcl':
        model,transform=ipclmodels.ipcl1()
    if type=='category':
        model,transform=ipclmodels.ipcl15()
    if type=='random':
        model,transform=ipclmodels.ipcl16()
    model.cuda()
    model.eval()
    
    return model,transform