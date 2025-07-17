import jax
from jax import numpy as jnp


def process_datamat_lst(datamat_lst: list,
                        padding_mask: jnp.array,
                        use_anc_emb: bool,
                        use_desc_emb: bool,
                        use_prev_align_info: bool):
    """
    select which embedding, then mask out padding tokens
    
    B: batch size
    L_align: length of alignment
    
    
    Arguments
    ----------
    datamat_lst : list[ArrayLike, ArrayLike, ArrayLike]
        > first array: ancestor embedding (B, L_align, H)
        > second array: descendant embedding (B, L_align, H)
        > third array: previous position alignment info (B, L_align, 1)
    
    padding_mask : ArrayLike, (B, L_align)
    
    use_anc_emb : bool
        > use ancestor embedding information to generate evolutionary 
          model parameters?
        
    use_desc_emb : bool
        > use descendant embedding information to generate evolutionary 
          model parameters?
    
    use_prev_align_info : bool
        > use previous position alignment label?
    
    Returns
    --------
    datamat : ArrayLike, (B, L_align, n*H + d*5)
        concatenated and padding-masked features
        > n=1, if only using ancestor embedding OR descendant embedding
        > n=2, if using both embeddings
        > d=1 if use_prev_align_info, otherwise 0
        
    masking_mat: ArrayLike, (B, L_align, n*H + d*5)
        location of padding in alignment
        > n=1, if only using ancestor embedding OR descendant embedding
        > n=2, if using both embeddings
        > d=1 if use_prev_align_info, otherwise 0
    """
    to_concat = []
    
    if use_anc_emb:
        to_concat.append( datamat_lst[0] )
    
    if use_desc_emb:
        to_concat.append( datamat_lst[1] )
    
    if use_prev_align_info:
        to_concat.append( datamat_lst[2] )
    
    # datamat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+5): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+5): use_anc_emb & use_anc_emb & use_prev_align_info
    datamat = jnp.concatenate( to_concat, axis = -1 ) 
    
    # masking_mat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+6): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+6): use_anc_emb & use_anc_emb & use_prev_align_info
    new_shape = (padding_mask.shape[0],
                 padding_mask.shape[1],
                 datamat.shape[2]) 
    masking_mat = jnp.broadcast_to(padding_mask[...,None], new_shape)
    del new_shape
    
    # datamat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+5): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+5): use_anc_emb & use_anc_emb & use_prev_align_info
    datamat = jnp.multiply(datamat, masking_mat)
    return datamat


###############################################################################
### helpers for reporting   ###################################################
###############################################################################
def write_matrix_to_npy(out_folder,
                        mat,
                        key):
    with open(f'{out_folder}/PARAMS-MAT_{key}.npy', 'wb') as g:
        np.save( g, mat )

def maybe_write_matrix_to_ascii(out_folder,
                                mat,
                                key):
    mat = jnp.squeeze(mat)
    if len(mat.shape) <= 2:
        np.savetxt( f'{out_folder}/ASCII_{key}.tsv', 
                    np.array(mat), 
                    fmt = '%.8f',
                    delimiter= '\t' )
