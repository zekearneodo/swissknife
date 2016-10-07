import logging
import h5py

logger = logging.getLogger('tables')


def unlimited_rows_data(group, table_name, data):
    """
    Create a table with no max shape, to append data forever
    :param group: h5py Group object. parent group
    :param table_name: str. name of the table
    :param data: np.array with initial data. Can be empty
    :return:
    """
    logger.info('Creating unbounded table {0} in group {1}'.format(group.name, table_name))
    try:
        table = group.create_dataset(table_name,
                                     shape=data.shape,
                                     dtype=data.dtype,
                                     maxshape=(None, None))
        table[:] = data

    except RuntimeError as e:
        if 'Name already exists' in str(e):
            logger.debug('Table {} already exists, appending the data'.format(table_name))
            table = group[table_name]
            append_rows(table, data)
        else:
            raise
    return table


def append_rows(dataset, new_data):
    '''
    Append rows to an existing table
    :param dataset: h5py Dataset object. Where to append the data
    :param new_data: array. An array to append
    :return:
    '''
    rows = dataset.shape[0]
    more_rows = new_data.shape[0]
    dataset.resize(rows + more_rows, axis=0)
    if dataset.size == (rows + more_rows):
        dataset[rows:] = new_data
    else:
        dataset[rows:, :] = new_data


def load_table_slice(table, row_list=None, col_list=None):
    """
    Loads a slice of a h5 dataset.
    It can load sparse columns and rows. To do this, it first grabs the smallest chunks that contains them.
    :param table: dataset of an h5 file.
    :param row_list: list of rows to get (int list)
    :param col_list: list of cols to get (int list)
    :return: np.array of size row_list, col_list with the concatenated rows, cols.
    """
    table_cols = table.shape[1]
    table_rows = table.shape[0]
    d_type = table.dtype

    col_list = np.arange(table_cols) if col_list is None else np.array(col_list)
    row_list = np.arange(table_rows) if row_list is None else np.array(row_list)

    raw_table_slice = np.empty([np.ptp(row_list) + 1, np.ptp(col_list) + 1], dtype=np.dtype(d_type))
    table.read_direct(raw_table_slice,
                      np.s_[np.min(row_list): np.max(row_list) + 1, np.min(col_list): np.max(col_list) + 1])
    # return raw_table_slice
    return raw_table_slice[row_list - np.min(row_list), :][:, col_list - np.min(col_list)]