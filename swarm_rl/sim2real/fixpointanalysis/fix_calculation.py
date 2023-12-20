import numpy as np
import torch

if __name__ == '__main__':
    np.random.seed(0)

    shift = 12

    model = torch.load(
        "/home/saz/GitHub/quadswarmsharif/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    # model = torch.load("/home/saz/GitHub/quadswarmsharif/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    # print(model)

    act_nn = {'action_parameterization.distribution_linear.weight': np.array(
        model['model']['action_parameterization.distribution_linear.weight'].cpu(), dtype=np.float32),
              'action_parameterization.distribution_linear.bias': np.array(
                  model['model']['action_parameterization.distribution_linear.bias'].cpu(), dtype=np.float32)}

    for k in model['model'].keys():
        if 'actor_encoder' in k.split('.'):
            act_nn[k] = np.array(model['model'][k].cpu(), dtype=np.float32)

    neighbor_nn_w = {0: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'],
                     1: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.weight']}

    neighbor_nn_b = {0: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'],
                     1: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.bias']}

    self_nn_w = {0: act_nn['actor_encoder.self_encoder.0.weight'],
                 1: act_nn['actor_encoder.self_encoder.2.weight']}

    self_nn_b = {0: act_nn['actor_encoder.self_encoder.0.bias'],
                 1: act_nn['actor_encoder.self_encoder.2.bias']}

    feedforward_nn_w = {0: act_nn['actor_encoder.feed_forward.0.weight'],
                        1: act_nn['action_parameterization.distribution_linear.weight']}

    feedforward_nn_b = {0: act_nn['actor_encoder.feed_forward.0.bias'],
                        1: act_nn['action_parameterization.distribution_linear.bias']}

    samps = 1000
    self_inp_global = np.array(np.random.rand(18, samps), dtype=np.float32)
    self_inp_global[0, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)
    self_inp_global[1, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)
    self_inp_global[2, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)

    self_inp_global[3, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)
    self_inp_global[4, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)
    self_inp_global[5, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)

    self_inp_global[6, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[7, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[8, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)

    self_inp_global[9, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[10, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[11, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)

    self_inp_global[12, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[13, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)
    self_inp_global[14, :] = np.array(np.random.rand(samps) * 2.0 - 1.0, dtype=np.float32)

    self_inp_global[15, :] = np.array(np.random.rand(samps) * 80.0 - 40.0, dtype=np.float32)
    self_inp_global[16, :] = np.array(np.random.rand(samps) * 80.0 - 40.0, dtype=np.float32)
    self_inp_global[17, :] = np.array(np.random.rand(samps) * 80.0 - 40.0, dtype=np.float32)

    neighb_inp_global = np.array(np.random.rand(6, samps), dtype=np.float32)
    neighb_inp_global[0, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)
    neighb_inp_global[1, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)
    neighb_inp_global[2, :] = np.array(np.random.rand(samps) * 20.0 - 10.0, dtype=np.float32)

    neighb_inp_global[3, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)
    neighb_inp_global[4, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)
    neighb_inp_global[5, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)

    neighbor_nn_w_fix = {0: (act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.weight']*2**shift).astype(np.int32)  ,
                         1: (act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.weight']*2**shift).astype(np.int32)}

    neighbor_nn_b_fix = {0: (act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.bias']*2**shift).astype(np.int32)  ,
                         1: (act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.bias']*2**shift).astype(np.int32)}

    self_nn_w_fix = {0: (act_nn['actor_encoder.self_encoder.0.weight']*2**shift).astype(np.int32)  ,
                     1: (act_nn['actor_encoder.self_encoder.2.weight']*2**shift).astype(np.int32)}

    self_nn_b_fix = {0: (act_nn['actor_encoder.self_encoder.0.bias']*2**shift).astype(np.int32)  ,
                     1: (act_nn['actor_encoder.self_encoder.2.bias']*2**shift).astype(np.int32)}

    feedforward_nn_w_fix = {0: (act_nn['actor_encoder.feed_forward.0.weight']*2**shift).astype(np.int32)  ,
                            1: (act_nn['action_parameterization.distribution_linear.weight']*2**shift).astype(np.int32)}

    feedforward_nn_b_fix = {0: (act_nn['actor_encoder.feed_forward.0.bias']*2**shift).astype(np.int32)  ,
                            1: (act_nn['action_parameterization.distribution_linear.bias']*2**shift).astype(np.int32)}

    max_err = 0

    for s in range(samps):
        self_inp    = self_inp_global[:,s]
        neighb_inp  = neighb_inp_global[:,s]

        self_inp_fix = (self_inp*2**shift).astype(np.int32)
        neighb_inp_fix   = (neighb_inp*2**shift).astype(np.int32)

        out_self_0_fix = np.zeros(16).astype(np.int32)
        for k in range(16):
            for u in range(18):
                out_self_0_fix[k] += ((self_nn_w_fix[0][k][u]*self_inp_fix[u])*2**-shift)
            out_self_0_fix[k] += self_nn_b_fix[0][k]
            out_self_0_fix[k] = np.maximum(out_self_0_fix[k], 0)


        #out_self_0_fix = np.maximum(self_nn_w_fix[0] @ self_inp_fix + self_nn_b_fix[0].reshape(self_nn_b_fix[0].shape[0], 1), 0)
        out_self_1_fix = np.zeros(16)
        for k in range(16):
            for u in range(16):
                out_self_1_fix[k] += int((self_nn_w_fix[1][k][u]*out_self_0_fix[u])*2**-shift)
            out_self_1_fix[k] += self_nn_b_fix[1][k]
            out_self_1_fix[k] = np.maximum(out_self_1_fix[k], 0)

        #out_self_1_fix = np.maximum(self_nn_w_fix[1] @ out_self_0_fix + self_nn_b_fix[1].reshape(16, 1), 0)

        out_neighb_0_fix = np.zeros(8)
        for k in range(8):
            for u in range(6):
                out_neighb_0_fix[k] += int((neighbor_nn_w_fix[0][k][u]*neighb_inp_fix[u])*2**-shift)
            out_neighb_0_fix[k] += neighbor_nn_b_fix[0][k]
            out_neighb_0_fix[k] = np.maximum(out_neighb_0_fix[k], 0)


        #out_neighb_0_fix = np.maximum(neighbor_nn_w_fix[0] @ neighb_inp_fix + neighbor_nn_b_fix[0].reshape(neighbor_nn_b_fix[0].shape[0], 1), 0)

        out_neighb_1_fix = np.zeros(8)
        for k in range(8):
            for u in range(8):
                out_neighb_1_fix[k] += int((neighbor_nn_w_fix[1][k][u] * out_neighb_0_fix[u]) * 2 ** -shift)
            out_neighb_1_fix[k] += neighbor_nn_b_fix[1][k]
            out_neighb_1_fix[k] = np.maximum(out_neighb_1_fix[k], 0)

        #out_neighb_1_fix = np.maximum(neighbor_nn_w_fix[1] @ out_neighb_0_fix + neighbor_nn_b_fix[1].reshape(8, 1), 0)

        inp_ff_fix = np.concatenate((out_self_1_fix, out_neighb_1_fix))

        out_ff_1_fix = np.zeros(32)
        for k in range(32):
            for u in range(24):
                out_ff_1_fix[k] += int((feedforward_nn_w_fix[0][k][u] * inp_ff_fix[u]) * 2 ** -shift)
            out_ff_1_fix[k] += feedforward_nn_b_fix[0][k]
            out_ff_1_fix[k] = np.maximum(out_ff_1_fix[k], 0)


        #out_ff_1_fix = np.maximum(feedforward_nn_w_fix[0] @ inp_ff_fix + np.expand_dims(feedforward_nn_b_fix[0], axis=1), 0)

        out_ff_2_fix = np.zeros(4)
        for k in range(4):
            for u in range(32):
                out_ff_2_fix[k] += int((feedforward_nn_w_fix[1][k][u] * out_ff_1_fix[u]) * 2 ** -shift)
            out_ff_2_fix[k] += feedforward_nn_b_fix[1][k]

        #out_ff_2_fix = feedforward_nn_w_fix[1] @ out_ff_1_fix + np.expand_dims(feedforward_nn_b_fix[1], axis=1)

        out_self_0 = np.maximum(self_nn_w[0] @ self_inp + self_nn_b[0], 0)
        out_self_1 = np.maximum(self_nn_w[1] @ out_self_0 + self_nn_b[1], 0)

        out_neighb_0 = np.maximum(neighbor_nn_w[0] @ neighb_inp + neighbor_nn_b[0], 0)
        out_neighb_1 = np.maximum(neighbor_nn_w[1] @ out_neighb_0 + neighbor_nn_b[1], 0)

        inp_ff = np.concatenate((out_self_1, out_neighb_1))
        out_ff_1 = np.maximum(feedforward_nn_w[0] @ inp_ff + feedforward_nn_b[0], 0)
        out_ff_2 = feedforward_nn_w[1] @ out_ff_1 + feedforward_nn_b[1]

        # print()
        # print(out_ff_2_fix*2**(-shift))
        # print(out_ff_2)

        loc_err = np.abs( np.sqrt( np.sum( ( out_ff_2 - out_ff_2_fix*2**(-shift) )**2 ) )  )
        if max_err < loc_err:
            max_err = loc_err

    print(max_err)


