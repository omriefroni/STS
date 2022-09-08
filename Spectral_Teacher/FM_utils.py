import torch
from Spectral_Teacher.surfmnet_loss import SURFMNetLoss




def fm_step(batch, feat1, feat2):
    """
    Functional Map step - calculate 2 mapping matrices C at the spectral space, and SURFMNet loss.  

    Inputs: 
    - batch dictionary contains: 
        B- batch, N - number of points, K - number of LBO vectors, last dim: 0 - source, 1 - target
        LBO data:
        * 'evects' - LBO eigenvectors - [B, N, K, 2] 
        * 'evals' - LBO eigenvalues - [B, K, 2]
        * 'a' - faces areas for normalization [B, N, 2] - OR: 'evects_trans' - normalized LBO eigenvectors. 

        Features (from the student): 
        D - features dimensions 
        * feat1 - per-point features of the source - [N, D]
        * feat2 - per-point features of the target - [N, D]


    """
    # Loss definition
    fm_loss = SURFMNetLoss()


    # extract variables from batch
    evects1 = batch['evects'][:, :, :, 0]
    evects2 = batch['evects'][:, :, :, 1]
    A1 = batch['a'][:, :, 0]
    A2 = batch['a'][:, :, 1]
    evals_1 = batch['evals'][:,:,0]
    evals_2 = batch['evals'][:,:,1]
    if hasattr(batch, 'evects_trans'):
        evects1_trans = batch['evects_trans'][:, :, :, 0]
        evects2_trans = batch['evects_trans'][:, :, :, 0]
    else:
        evects1_trans = evects1.permute((0,2,1)) * A1[:,None,:]
        evects2_trans = evects2.permute((0,2,1)) * A2[:,None,:]

    f_hat = torch.bmm(evects1_trans, feat1)
    g_hat = torch.bmm(evects2_trans, feat2)
    f_hat, g_hat = f_hat.transpose(1, 2), g_hat.transpose(1, 2)

    C1 = []
    C2 = []
    # P1 = []
    # P2 = []

    for i in range(feat1.shape[0]):
        fm1 = torch.matmul(feat1.transpose(2, 1)[i], evects1_trans.transpose(2, 1)[i])
        fm2 = torch.matmul(feat2.transpose(2, 1)[i], evects2_trans.transpose(2, 1)[i])


        fm1_inv = fm1.transpose(1,0).pinverse()

        c_mat = torch.matmul(fm2.transpose(1,0), fm1_inv)
        C1.append(c_mat.unsqueeze(0))

        # if is_plot:
        #     p1 = FM_to_p2p(c_mat, evects1[i], evects2[i])
        #     P1.append(p1)

        fm2_inv = fm2.transpose(1,0).pinverse()
        c_mat = torch.matmul(fm1.transpose(1,0), fm2_inv)
        C2.append(c_mat.unsqueeze(0))

        # if is_plot:
        #     p2 = FM_to_p2p(c_mat, evects2[i], evects1[i])
        #     P2.append(p2)

    C1 = torch.cat(C1, dim=0)
    C2 = torch.cat(C2, dim=0)

    curr_loss = fm_loss(C1, C2, feat1, feat2, evects1, evects2,  evects1_trans, evects2_trans, evals_1, evals_2, C1.device )
    return curr_loss

