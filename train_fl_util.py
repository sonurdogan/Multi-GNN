from data_loading import get_fl_data
from train_util import AddEgoIds, add_arange_ids, get_loaders

def get_fl_loaders(partition_id: int, args, data_config):

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_fl_data(partition_id, args, data_config)

    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    return tr_loader, val_loader, te_loader