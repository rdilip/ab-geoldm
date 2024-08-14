import torch
import torch.nn.functional as F

def batched_diffab_to_geoldm(data, num_neighbors=50):
    """ Hack to turn data from diffab to geoldm """
    one_hot = F.one_hot(data['aa'], 22)
    atom_mask = data['aa'] != 21 

    one_hot = one_hot * (atom_mask.to(int).unsqueeze(-1))
    cdr_mask = data['cdr_flag']
    positions = data['pos_heavyatom'][:, :, 1, :] # index 1 is carbon-alpha atom
    num_atoms = atom_mask.sum(-1)

    # masked positions should have zero norm
    if not torch.all(atom_mask):
        assert positions.norm(dim=-1).masked_select(~atom_mask).abs().max() == 0.

    # Compute edge mask -- num_neighbors across amino acid positions
    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]

    D = (positions[:, :, None, :] - positions[:, None, :, :]).norm(dim=-1)
    # skip self. we can also just set the diagonal off on the pair mask
    D.masked_fill(~pair_mask, torch.inf)
    nn = D.argsort(dim=-1)[:, :, 1:num_neighbors+1] 
    
    edge_mask = torch.zeros_like(pair_mask)
    edge_mask = edge_mask.scatter_(-1, nn, 1)


    return dict(
        one_hot=one_hot,
        atom_mask=atom_mask,
        positions=positions,
        num_atoms=num_atoms,
        edge_mask=edge_mask,
        cdr_mask=cdr_mask,
        charges=None,
        generate_flag = data['generate_flag'].unsqueeze(-1)
    )
    
