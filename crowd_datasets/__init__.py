# build dataset according to given 'dataset_file'
# from torch.utils.data import DataLoader


def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
    elif args.dataset_file=='UCFCC':
        from crowd_datasets.UCFCC50.loading_data import loading_data
        return loading_data
    return None

