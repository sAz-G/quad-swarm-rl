import numpy as np
import torch

if __name__ == '__main__':
    inp_pth = '/home/saz/Desktop/qsrl_test/train_dir/mean_embed_16_8_relu/checkpoint_p0/best_000602426_616884224_reward_1.538.pth'
    model = torch.load(inp_pth, map_location='cpu')

    actor  = {}
    modelKeys = model['model'].keys()

    for k in modelKeys:
        if 'actor_encoder' in k.split('.'):
            actor[k] = model['model'][k].detach().cpu().clone().numpy()
            actor[k] = np.array(actor[k], dtype=float)

    actor['action_parameterization.distribution_linear.weight'] = model['model'][
        'action_parameterization.distribution_linear.weight'].detach().cpu().clone().numpy()
    actor['action_parameterization.distribution_linear.bias'] = model['model'][
        'action_parameterization.distribution_linear.bias'].detach().cpu().clone().numpy()

    actor['action_parameterization.distribution_linear.weight'] = actor[
        'action_parameterization.distribution_linear.weight'].astype(dtype=float)
    actor['action_parameterization.distribution_linear.bias'] = actor[
        'action_parameterization.distribution_linear.bias'].astype(dtype=float)

    out_pth = '/home/saz/Desktop/qsrl_test/csv_arrs/'
    for arr in actor:
        #np.savetxt(out_pth + k + '.txt', str(actor[k]))

        with open(out_pth + k + '.txt', "w") as txt_file:
            #txt_file.write("[")
            for arr in actor[k]:
                txt_file.write('{')
                if arr.size > 1:
                    for row in arr:
                        txt_file.write('{')
                        for val in arr:
                            txt_file.write(str(val) + ',')  # works with any number of elements in a line
                        txt_file.write('},')
                    txt_file.write('}')
                else:
                    for val in actor[k]:
                        txt_file.write(str(val) + ',')  # works with any number of elements in a line
                txt_file.write('},')




