import utils.models as models
from PIL import Image
import numpy as np
from fastprogress.fastprogress import progress_bar
import gc
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
def create_activation_dict(images,model_name,downscale='SRP'):
    activations = {}
    def get_activations(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    activationdict={}
    layers_list=[]
    
    

                
    # Decide which layers to extract
    
    if model_name.startswith("ipcl_"):
        model_names=model_name.split('_')
        model,transform=models.IPCLMODEL(type=model_names[1])
        for name, child in model.named_children():
            child.register_forward_hook(get_activations(name))
            activationdict[name]=[]
            layers_list.append(name)
            
    if model_name=='alexnet':
        model,transform=models.ALEXNET()
        for name, child in model.features.named_children():
            child.register_forward_hook(get_activations('Features_'+name))
            activationdict['Features_'+name]=[]
            layers_list.append('Features_'+name)
            
        for name, child in model.named_children():
            if name=='avgpool':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
        
        for name, child in model.classifier.named_children():
            if name!='0' and name!='3':
                child.register_forward_hook(get_activations('Classifier_'+name))
                activationdict['Classifier_'+name]=[]
                layers_list.append('Classifier_'+name)

            
    if model_name.startswith("slip_"):
        model_names=model_name.split('_')
        model,transform=models.SLIPMODELS(type=model_names[1],size=model_names[2])
        
        for name, child in model.named_children():
            if name=='patch_embed':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
        for name, child in model.blocks.named_children():
            child.register_forward_hook(get_activations(name))
            activationdict[name]=[]
            layers_list.append(name)
        for name, child in model.named_children():
            if name=='fc_norm':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
                

    
                    
    if model_name.startswith('VGG_'):
        model_names=model_name.split('_')
        model,transform=models.VGG(size=model_names[1])
        for name, child in model.features.named_children():
            child.register_forward_hook(get_activations('Features_'+name))
            activationdict['Features_'+name]=[]
            layers_list.append('Features_'+name)
            
        for name, child in model.named_children():
            if name=='avgpool':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
        
        for name, child in model.classifier.named_children():
            if name!='2' and name!='4' and name!='5' and name!='6':
                child.register_forward_hook(get_activations('Classifier_'+name))
                activationdict['Classifier_'+name]=[]
                layers_list.append('Classifier_'+name)
        
    if model_name.startswith('resnet_'):
        model_names=model_name.split('_')
        model,transform=models.RESNET(size=model_names[1])
        for name, child in model.named_children():
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
                
    if model_name.startswith('VIT_'):
        model_names=model_name.split('_')
        model,transform=models.VIToriginal(size=model_names[1])
        for name, child in model.named_children():
            if name=='conv_proj':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
            
        for name, child in model.encoder.layers.named_children():
            child.register_forward_hook(get_activations(name))
            activationdict[name]=[]
            layers_list.append(name)

        for name, child in model.encoder.named_children():
            if name=='ln':
                child.register_forward_hook(get_activations('Classifier_'+name))
                activationdict['Classifier_'+name]=[]
                layers_list.append('Classifier_'+name)
                
        for name, child in model.named_children():
            if name=='heads':
                child.register_forward_hook(get_activations(name))
                activationdict[name]=[]
                layers_list.append(name)
                
    if model_name.startswith('TDANN_'):
        model_names=model_name.split('_')
        model,transform=models.TDANN(type=model_names[1],lw=int(model_names[2]))
    
        for name, child in model.layer1.named_children():
            child.register_forward_hook(get_activations('layer1.'+name))
            activationdict['layer1.'+name]=[]
            layers_list.append('layer1.'+name)
        for name, child in model.layer2.named_children():
            child.register_forward_hook(get_activations('layer2.'+name))
            activationdict['layer2.'+name]=[]
            layers_list.append('layer2.'+name)
        for name, child in model.layer3.named_children():
            child.register_forward_hook(get_activations('layer3.'+name))
            activationdict['layer3.'+name]=[]
            layers_list.append('layer3.'+name)
        for name, child in model.layer4.named_children():
            child.register_forward_hook(get_activations('layer4.'+name))
            activationdict['layer4.'+name]=[]
            layers_list.append('layer4.'+name)
    
        
        
    #Extract and do dimensionality reduction if needed
    layersatatime=15

    print(layers_list)
    for kk in range(0, len(layers_list), layersatatime):
        for image in progress_bar(images):
            img = Image.fromarray(np.uint8(image*255)).convert('RGB')
            input = transform(img).unsqueeze_(0).cuda()
            output = model(input) 
            for i in layers_list[kk:kk+layersatatime]:
                activationdict[i].append(activations[i][0].detach().cpu().numpy()) 
        
        for i in layers_list[kk:kk+layersatatime]:  
            X=np.array(activationdict[i]).reshape(images.shape[0],-1)
            X = X[:, X.sum(0)!=0]
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0
            if downscale=='SRP':
                n_projections = johnson_lindenstrauss_min_dim(X.shape[0], eps=.1)
                srp = SparseRandomProjection(n_projections, random_state=42)
                activationdict[i]=srp.fit_transform(X)
                print(activationdict[i].shape)
                del srp
            else:
                activationdict[i]=X
                print(activationdict[i].shape)            
    

    print(list(activationdict.keys()))    
        

            

    del model
    del transform
    del input
    gc.collect()
    return activationdict
