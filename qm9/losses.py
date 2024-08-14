import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    node_mask = (node_mask > 0).to(int)
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, generate_flag=None):
    """ Generate flag allows us ot not try to regenerate the entire thing so we can provide a framework region
    """
    bs, n_nodes, n_dims = x.size()

    if generate_flag is None:
        generate_flag = node_mask


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context, generate_flag=generate_flag)


        # we only do diffusion on the masked CDRs.
        N = (node_mask * generate_flag).squeeze(2).sum(1).long()

        try:
            log_pN = nodes_dist.log_prob(N)
        except:
            breakpoint()

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
