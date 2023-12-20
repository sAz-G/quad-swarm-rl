import torch
from swarm_rl.models.quad_multi_model import QuadMultiEncoder
if __name__ == '__main__':

    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    #model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/mean_embed_16_8_relu_relpos/checkpoint_p0/checkpoint_000000003_3072.pth")
    #model = torch.load(
    #    "/train_dir/mean_embed_32_32_relu_tpos__tanhatend/checkpoint_p0/best_000379944_389062656_reward_2.071.pth")
#    model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/test_multi_drone/checkpoint_p0/checkpoint_000000003_3072.pth")
    model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/mean_embed_32_32_tanh/checkpoint_p0/best_000001220_1249280_reward_-30.502.pth")

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