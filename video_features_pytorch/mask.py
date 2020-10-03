import torch


def perturb_sequence(seq, mask, perturbation_type='freeze', snap_values=False):
    if snap_values:
        for j in range(len(mask)):
            if mask[j] > 0.5:
                mask[j] = 1
            else:
                mask[j] = 0
    if perturbation_type == 'freeze':
        # Pytorch expects Batch, Channel, T, H, W
        perturbed_input = torch.zeros_like(seq)  # seq.clone().detach()
        # TODO: also double check that the mask here is not a copy but by ref so the grad holds
        for u in range(len(mask)):

            if u == 0:
                perturbed_input[:, :, u, :, :] = seq[:, :, u, :, :]
            if u != 0:
                perturbed_input[:, :, u, :, :] = (1 - mask[u]) * \
                                                 seq[:, :, u, :, :] + \
                                                 mask[u] * perturbed_input.clone()[:, :, u - 1, :, :]

    if perturbation_type == 'reverse':
        # pytorch expects Batch,Channel, T, H, W
        perturbed_input = torch.zeros_like(seq)

        # threshold for finding out which indexes are on
        mask_on_inds = (mask > 0.1).nonzero()
        if len(mask_on_inds) > 0:
            # non-zero gives unfortunate dimensions like [[0], [1]] so squeeze it
            mask_on_inds = mask_on_inds.squeeze(dim=1)
        mask_on_inds = mask_on_inds.tolist()

        submasks = find_submasks_from_mask(mask, 0.1)

        for y in range(len(mask)):
            # start with original
            perturbed_input[:, :, y, :, :] = seq[:, :, y, :, :]

        for mask_on_inds in submasks:
            '''if the submask is on at 3,4,5,6,7. the point is to go to the almost halfway point
            as in 3,4 (so that's why it's len()//2) and swap it with the u:th last element instead (like 7,6)
            '''
            for u in range(int(len(mask_on_inds) // 2)):
                # temp storage when doing swap
                temp = seq[:, :, mask_on_inds[u], :, :].clone()
                # first move u:th LAST frame to u:th frame
                perturbed_input[:, :, mask_on_inds[u], :, :] = \
                    (1 - mask[mask_on_inds[u]]) * seq[:, :, mask_on_inds[u], :, :] + \
                    mask[mask_on_inds[u]] * seq[:, :, mask_on_inds[-(u + 1)], :, :]

                # then use temp storage to move u:th frame to u:th last frame
                perturbed_input[:, :, mask_on_inds[-(u + 1)], :, :] = \
                    (1 - mask[mask_on_inds[u]]) * seq[:, :, mask_on_inds[-(u + 1)], :, :] + \
                    mask[mask_on_inds[u]] * temp
    return perturbed_input


def find_submasks_from_mask(mask, thresh=0.1):
    submasks = []
    current_submask = []
    currently_in_mask = False

    for j in range(len(mask)):

        # if we find a value above threshold but is first occurence, start appending to current submask
        if mask[j] > thresh and not currently_in_mask:
            current_submask = []
            currently_in_mask = True
            current_submask.append(j)
        # if it's not current occurence, just keep appending
        elif mask[j] > thresh and currently_in_mask:
            current_submask.append(j)

        # if below thresh, stop appending
        elif mask[j] <= thresh and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False

        # if at the end of clip, create last submask
        if j == len(mask) - 1 and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False
    return submasks


def calc_tv_norm(mask, p=3, q=3):
    '''
    Calculates the Total Variational Norm by summing the differences of the values in between the different positions
    in the mask. p=3 and q=3 are defaults from the paper.
    '''
    val = 0
    for u in range(1, len(mask) - 1):
        val += torch.abs(mask[u - 1] - mask[u]) ** p
        val += torch.abs(mask[u + 1] - mask[u]) ** p
    val = val ** (1 / p)
    val = val ** q

    return val


def init_mask(seq, model, batch_index, target,
              threshold=0.9, mode='central', mask_type='freeze'):
    """
    Initiaizes the mask. Central mode finds the smallest centered
    mask which still reduces the class score by at least 90% compared
    to a fully perturbing mask (whole mask on).
    Random mode turns on 70% of the mask (on average).

    :param seq: torch.tensor
    :param model:
    :param batch_index: int
    :param target:
    :param threshold:
    :param mode: str [random|central]
    :param mask_type: str ['freeze|reverse']
    :return: torch.tensor
    """

    if mode == "central":

        fully_frozen_seq = torch.zeros_like(seq)
        # first define the fully perturbed sequence
        for i in range(seq.shape[2]):
            fully_frozen_seq[:, :, i, :, :] = seq[:, :, 0, :, :]

        fully_frozen_score = model(fully_frozen_seq)[batch_index, target[batch_index]]
        orig_score = model(seq)[batch_index, target[batch_index]]

        mask = torch.cuda.FloatTensor([1] * seq.shape[2])

        # reduce mask size while the loss ratio remains above 90%
        for i in range(1, seq.shape[2] // 2):
            new_mask = torch.cuda.FloatTensor([1] * seq.shape[2])
            new_mask[:i] = 0
            new_mask[-i:] = 0

            central_score = model(
                perturb_sequence(seq, new_mask, perturbation_type=mask_type)
            )[batch_index, target[batch_index]]
            score_ratio = (orig_score - central_score) / (orig_score - fully_frozen_score)

            if score_ratio < threshold:
                break

        mask = new_mask

        # modify the mask so that it is roughly 0 or 1 after sigmoid
        for j in range(len(mask)):
            if mask[j] == 0:
                mask[j] = -5
            elif mask[j] == 1:
                mask[j] = 5

    elif mode == "random":
        # random init to 0 or 1, then modify for sigmoid
        mask = torch.cuda.FloatTensor(seq.shape[2]).uniform_() > 0.7
        mask = mask.float()
        mask = mask - 0.5
        mask = mask * 5

        # if mask were to be ALL 0's or 1's, perturb one a bit so that TV norm doesn't NaN
        if torch.abs(mask.sum()) == 2.5 * len(mask):
            mask[8] += 0.1

    mask.requires_grad_()
    print("initial mask is: ", mask)
    return mask


def freezeFrame(seq, mask, u):
    '''
    A recursive way of calculating the frozen frames, might be required if the framework does not allow
    'self assign' via the .clone() operation (requires A LOT of memory though). Meant to return the value
    of the frozen frame at position 'u'. Not used as of now.
    '''
    if u == 0:
        return seq[:, :, u, :, :]
    if u != 0:  # mask[u]>=0.5 and u!=0
        return (((1 - mask[u]) * seq[:, :, u, :, :] + mask[u] * freezeFrame(seq, mask, u - 1)) / (
            (1 - mask[u]) * seq[:, :, u, :, :] + mask[u] * freezeFrame(seq, mask, u - 1)).max()) * 255


