import torch
from swarm_rl.models.quad_multi_model import QuadMultiEncoder
if __name__ == '__main__':

    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8(2)/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    model = torch.load("/train_dir/attention_16_8_relu(2)/checkpoint_p0/best_000550010_563210240_reward_0.712.pth", map_location ='cpu')
    #print(model['model'].keys())
    #print(model['model'])
    #'action_parameterization.distribution_linear.weight'
    dictio =  {'action_parameterization.distribution_linear.weight' : model['model']['action_parameterization.distribution_linear.weight'], 'action_parameterization.distribution_linear.bias':model['model']['action_parameterization.distribution_linear.bias']}

    for k in model['model'].keys():
        if 'actor_encoder' in k.split('.'):
            dictio[k] = model['model'][k]

    for v in dictio.values():
        print(v.shape)


    print(dictio.keys())
    print(dictio['action_parameterization.distribution_linear.bias'])