import torch 

from pearl.src.methods.global_infonce_stdim import CLIPGlobalInfoNCESpatioTemporalTrainer
from pearl.src.methods.global_local_infonce import CLIPGlobalLocalInfoNCESpatioTemporalTrainer, CLIPGlobalLocalInfoNCESpatialTrainer
from pearl.src.methods.stdim import CLIPInfoNCESpatioTemporalTrainer, CLIPInfoNCESpatioFullTrainer
from pearl.src.methods.cpc_clip import CLIPCPCTrainer

def run_encoder_training(encoder, tr_eps, val_eps, config, wandb, method="global-infonce-stdim", pretrained_encoder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add different training methods here
    if method == "global-infonce-stdim": 
        trainer = CLIPGlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif method == 'clip-cpc':
        trainer = CLIPCPCTrainer(encoder, config, device=device, wandb=wandb)
    elif method == "global-local-spatial-infonce":
        trainer = CLIPGlobalLocalInfoNCESpatialTrainer(encoder, config, device=device, wandb=wandb)
    elif method == "stdim":
        trainer = CLIPInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif method == "sdim":
        trainer = CLIPInfoNCESpatioFullTrainer(encoder, config, device=device, wandb=wandb)
    else:
        raise Exception("Invalid method...please pick a valid encoder training method")

    if pretrained_encoder:
        trainer.encoder = pretrained_encoder

    trainer.train(tr_eps, val_eps)

    return encoder
