import numpy as np
import torch
from swarm_rl.models.quad_multi_model import QuadMultiEncoder
if __name__ == '__main__':

    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    model = torch.load("/home/saz/GitHub/quadswarmsharif/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    #model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/mean_embed_16_8_relu_relpos/checkpoint_p0/checkpoint_000000003_3072.pth")
#    model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/test_multi_drone/checkpoint_p0/checkpoint_000000003_3072.pth")

    #print(model['model'].keys())
    #print(model['model'])
    #'action_parameterization.distribution_linear.weight'
    dictio =  {'action_parameterization.distribution_linear.weight' : np.array(model['model']['action_parameterization.distribution_linear.weight'].cpu(), dtype=np.float32),
               'action_parameterization.distribution_linear.bias':np.array(model['model']['action_parameterization.distribution_linear.bias'].cpu(), dtype=np.float32)}


    for k in model['model'].keys():
        if 'actor_encoder' in k.split('.'):
            dictio[k] = np.array(model['model'][k].cpu(), dtype=np.float32)

    max_weight_abs = 0
    for v in dictio.values():
        if max_weight_abs < np.max(v):
            max_weight_abs = np.max(v)




    print(max_weight_abs)



