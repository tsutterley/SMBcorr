import SMBcorr.utilities

# attempt imports
h5py = SMBcorr.utilities.import_dependency('h5py')

def get_h5_structure(h5_file, max_depth):
    '''
    get the subdirectory structure of file to specified depth
    '''

    sub_list=[]
    last_depth=0
    with h5py.File(h5_file,'r') as h5f:
        get_h5_groups(h5f, '', sub_list, last_depth, max_depth)
    return sub_list


def get_h5_groups(h5f, key, sub_list, last_depth, max_depth):
    '''
    helper function to recursively map group structure of an hdf5 file
    '''
    this_depth=last_depth+1
    temp_key=key
    if len(temp_key)==0:
        temp_key='/'
    subs=[key+'/'+new_key for new_key in h5f[temp_key].keys()]
    if this_depth ==max_depth:
        sub_list += subs
        return
    else:
        for new_key in subs:
            get_h5_groups(h5f, new_key, sub_list, this_depth, max_depth)
