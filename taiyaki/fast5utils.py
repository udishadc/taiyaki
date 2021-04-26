# Utilities to read and write information from HDF5 files,
# including ONT fast5 files.
# ONT fast5 access is built on top of the ont_fast5_api
import os
import sys
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
from ont_fast5_api.fast5_interface import get_fast5_file
from taiyaki.fileio import readtsv


##################################################
# FUNCTIONS TO ITERATE OVER READS IN FAST5 FILES #
##################################################

def iterate_file_read_pairs(filepaths, read_ids, limit=None, verbose=0):
    """Iterate over file and read id pairs.

    Args:
        filepaths (list of str): list of filepaths
        read_ids (list of str): list of read ids
        limit (int, optional): Maximum number of tuples to produce
        verbose (int, optional): Output level of debug verbosity

    Yields:
        tuple(str, str): containing filepath and read id
    """
    nyielded = 0
    for filepath, read_id in zip(filepaths, read_ids):
        if not os.path.exists(filepath):
            sys.stderr.write(
                'File {} does not exist, skipping\n'.format(filepath))
            continue
        try:
            with get_fast5_file(filepath, 'r') as f5file:
                if read_id not in f5file.get_read_ids():
                    continue
            if verbose > 0:
                print("Reading", read_id, "from", filepath)
            yield filepath, read_id
            nyielded += 1
            if limit is not None and nyielded >= limit:
                return  # ends iterator
        except Exception as e:
            sys.stderr.write((
                "Warning: An exception occured in fast5utils (skipped " +
                "this read):\n{}\n").format(str(e)))
    return


def iterate_files_reads_unpaired(filepaths, read_ids, limit=None, verbose=0):
    """ Iterate over unpaired lists of filepaths and read_ids, looking in all
    the files given and returning only those read_ids in the read_ids list. If
    read_ids is None then get all the reads in the files.

    Args:
        filepaths (list of str): list of filepaths
        read_ids (list of str or None): list of read ids or None for all reads
            in each file
        limit (int, optional): Maximum number of tuples to produce
        verbose (int, optional): Output level of debug verbosity

    Yields:
        tuple(str, str): containing filepath and read id
    """
    nyielded = 0
    for filepath in filepaths:
        if not os.path.exists(filepath):
            sys.stderr.write(
                'File {} does not exist, skipping\n'.format(filepath))
            continue
        try:
            with get_fast5_file(filepath, 'r') as f5file:
                for read_id in f5file.get_read_ids():
                    if read_ids is None or read_id in read_ids:
                        if verbose > 0:
                            print("Reading", read_id, "from", filepath)
                        yield filepath, read_id
                        nyielded += 1
                    else:
                        if verbose > 0:
                            print("Skipping", read_id, "from",
                                  filepath, ":not in read_id list")
                    if limit is not None and nyielded >= limit:
                        return  # ends iterator
        except Exception as e:
            sys.stderr.write((
                "Warning: An exception occured in fast5utils " +
                "(skipped this read):\n{}\n").format(str(e)))


def iterate_fast5_reads(
        path, strand_list=None, limit=None, verbose=0, recursive=False):
    """ Iterate over reads in a directory of fast5 files or a single fast5
        file. Files may be single or multi-read fast5s.

    Args:
        path (str): Directory (or filename for a single file)
        strand_list (str or None, optional): Path to file containing list of
            files and/or read ids to iterate over (as described in notes) or
            None for all files and reads
        limit (int or None, optional): Maximum number of reads to consider or
            None for all
        verbose (int, optional): 0 prints no messages, 1 prints a message for
            every file read, 2 prints the list of files before starting as well
        recursive (bool, optional): Search path recursively for fast5 files

    Yields:
        (tuple(str, str)): filepath and read_id for each read. You may say,
            "why not yield an ont_fast_api object instead of a nasty tuple?" I
            would say: "yes, I tried that, but it led to unfathomable nastiness
            when I fed these objects in as arguments to multiple processes."

    Notes:
        If strand_list is given, then only return the reads spcified, according
        to the following rules:

        (A) If the strand list file has a column 'read_id' and no column
            'filename' or 'filename_fast5' then look through all fast5 files in
            the path and return all reads with read_ids in that column.
        (B) If the strand list file has a column 'filename' or 'filename_fast5'
            and no column 'read_id' then look through all filenames specified
            and return all reads in them.
        (C) If the strand list has a column 'filename' or 'filename_fast5'
            _and_ a column 'read_id' then loop through the rows in the strand
            list, returning the appropriate tuple for each row. We check that
            each file exists and contains the read_id.

    Example:
        read_iterator = iterate_fast5_reads('directory')
        for read_tuple in read_iterator:
            fname,read_id = read_tuple
            print("Filename=",fname,", read id = ",read_id)
            with fast5_interface.get_fast5_file(fname, 'r') as f5file:
                read = f5file.get_read(read_id)
                dacs = read.get_raw_data()
            print("Length of rawget_file_names data:",len(dacs))
    """
    filepaths, read_ids = None, None

    if strand_list is not None:
        strand_table = readtsv(strand_list)
        if verbose >= 2:
            print("Columns in strand list file:")
            print(strand_table.dtype.names)
        if 'filename' in strand_table.dtype.names:
            filepaths = strand_table['filename']
        elif 'filename_fast5' in strand_table.dtype.names:
            filepaths = strand_table['filename_fast5']
        if 'read_id' in strand_table.dtype.names:
            read_ids = [str(i) for i in strand_table['read_id']]
        # If we get to this point and we haven't got read ids or filenames,
        # then there is nothing in the strand list that we can use (this
        # happens, for example, when the strand list has no header line).
        if filepaths is None and read_ids is None:
            raise Exception((
                "Strand list at {} has no column that can be used: (it " +
                "should contain ('filename' or 'filename_fast5') or " +
                "'read_id', or both a filename column and a read_id " +
                "column)").format(strand_list))
        # The strand list supplies filenames, not paths, so we supply the rest
        # of the path
        if filepaths is not None:
            filepaths = [os.path.join(path, x) for x in filepaths]

    if (filepaths is not None) and (read_ids is not None):
        # This is the case (C) above. Both filenames and read_ids come from the
        # strandlist and we therefore know which read_id goes with which file
        for y in iterate_file_read_pairs(filepaths, read_ids, limit, verbose):
            yield y
        return

    if filepaths is None:
        # Filenames not supplied by strand list, so we get them from the path
        if os.path.isdir(path):
            filepaths = get_fast5_file_list(path, recursive=recursive)
        else:
            filepaths = [path]

    for y in iterate_files_reads_unpaired(filepaths, read_ids, limit, verbose):
        yield y


######################################################
# FUNCTIONS TO READ INFORMATION FROM ONT FAST5 FILES #
######################################################

def get_filename(read):
    """ Get filename from read object

    Args:
        read (:ont_fast5_api:`Fast5Read`): the read object

    Returns:
        str: filename

    Examples:
        from ont_fast5_api import fast5_interface

        #SINGLE READ
        s5 = fast5_interface.get_fast5_file(singleReadFile, 'r')
        read_id = s5.get_read_ids()[0]
        read = s5.get_read(read_id)
        get_filename(read)

        #MULTI-READ
        m5 = fast5_interface.get_fast5_file(multiReadFile, 'r')
        for nread,read_id in enumerate(m5.get_read_ids()):
            read = m5.get_read(read_id)
            get_filename(read)
    """
    return read.handle[read.global_key + 'context_tags'].attrs['filename']


def get_channel_info(read):
    """Get channel info from read object.

    Args:
        read (:ont_fast5_api:`Fast5Read`): the read object

    Returns:
        dict: channel info including digitisation, range, offset, sampling_rate

    Examples:
        from ont_fast5_api import fast5_interface

        #SINGLE READ
        s5 = fast5_interface.get_fast5_file(singleReadFile, 'r')
        read_id = s5.get_read_ids()[0]
        read = s5.get_read(read_id)
        get_channel_info(read)

        #MULTI-READ
        m5 = fast5_interface.get_fast5_file(multiReadFile, 'r')
        for nread,read_id in enumerate(m5.get_read_ids()):
            read = m5.get_read(read_id)
            get_channel_info(read)
    """
    # This is how it is done in _load_raw() in AbstractFast5File in
    # ont_fast5_api.fast5_file.py
    return read.handle[read.global_key + 'channel_id'].attrs


def get_read_attributes(read):
    """Get read attributes from read object.

    Args:
        read (:ont_fast5_api:`Fast5Read`): the read object

    Returns:
        dict: read attributes including start_time, read_id, duration

    Examples:
        from ont_fast5_api import fast5_interface

        #SINGLE READ
        s5 = fast5_interface.get_fast5_file(singleReadFile, 'r')
        read_id = s5.get_read_ids()[0]
        read = s5.get_read(read_id)
        get_read_attributes(read)

        #MULTI-READ
        m5 = fast5_interface.get_fast5_file(multiReadFile, 'r')
        for nread,read_id in enumerate(m5.get_read_ids()):
            read = m5.get_read(read_id)
            get_read_attributes(read)
    """
    # In a multi-read file, they should be here...
    r = read.handle['Raw'].attrs
    if len(r) > 0:
        return r
    # In a single-read file, they are here...
    # We want the highest numbered read (latest)
    # in the tree 'Raw/Reads/Read_XXXX'
    # where XXXX is a number like 0021 or 0001
    numbered_reads = list(read.handle['Raw/Reads'].keys())
    last_numbered_read = sorted(numbered_reads)[-1]
    return read.handle['Raw/Reads/' + last_numbered_read].attrs


def read_summary(read):
    """Print summary of information in fast5 file on a particular read object.

    Args:
        read (:ont_fast5_api:`Fast5Read`): the read object

    Examples:
        from ont_fast5_api import fast5_interface

        #SINGLE READ
        s5 = fast5_interface.get_fast5_file(singleReadFile, 'r')
        read_id = s5.get_read_ids()[0]
        read = s5.get_read(read_id)
        read_summary(read)

        #MULTI-READ
        m5 = fast5_interface.get_fast5_file(multiReadFile, 'r')
        for nread,read_id in enumerate(m5.get_read_ids()):
            read = m5.get_read(read_id)
            read_summary(read)
    """
    print("ONT interface: read information")
    dacs = read.get_raw_data()
    channel_info = get_channel_info(read)
    read_attributes = get_read_attributes(read)
    print("     signal data =", dacs[:10], '...')
    print("     signal metadata: channel info")
    for k, v in channel_info.items():
        print("           ", k, v)
    print("     signal metadata: read attributes")
    for k, v in read_attributes.items():
        print("           ", k, v)
