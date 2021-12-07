#!/usr/bin/env python
u"""
ftp_mar_data.py
Written by Tyler Sutterley (10/2021)

Syncs MAR regional climate outputs for a given ftp url
    ftp://ftp.climato.be/fettweis

CALLING SEQUENCE:
    python ftp_mar_data.py --directory <path> <ftp://url>

INPUTS:
    full ftp url

COMMAND LINE OPTIONS:
    --help: list the command line options
    -P X, --np X: Run in parallel with X number of processes
    -D X, --directory X: full path to working data directory
    -Y X, --year X: Reduce files to years of interest
    -C, --clobber: Overwrite existing data in transfer
    -T X, --timeout X: Timeout in seconds for blocking operations
    -l, --log: output log of files downloaded
    -M X, --mode X: Local permissions mode of the directories and files synced

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 10/2021: using python logging for handling verbose output
        use utilities module for finding and retrieving files
    Updated 05/2020: added years option to reduce list of files
    Updated 11/2019: added multiprocessing option to run in parallel
    Written 07/2019
"""
from __future__ import print_function

import sys
import os
import time
import logging
import argparse
import traceback
import posixpath
import multiprocessing
import SMBcorr.utilities

#-- PURPOSE: sync local MAR files with ftp and handle error exceptions
def ftp_mar_data(parsed_ftp, DIRECTORY=None, YEARS=None, TIMEOUT=None,
    LOG=False, PROCESSES=0, CLOBBER=False, MODE=0o775):

    #-- check if local directory exists and recursively create if not
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY,mode=MODE)

    #-- output of synchronized files
    if LOG:
        #-- format: MAR_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = 'MAR_sync_{0}.log'.format(today)
        logging.basicConfig(filename=os.path.join(DIRECTORY,LOGFILE),
            level=logging.INFO)
        logging.info('ICESat-2 Data Sync Log ({0})'.format(today))

    else:
        #-- standard output (terminal output)
        logging.basicConfig(level=logging.INFO)

    #-- list directories from ftp
    R1 = r'\d+' if not YEARS else r'|'.join(map(str,YEARS))
    # find files and reduce to years of interest if specified
    remote_files,remote_times = SMBcorr.utilities.ftp_list(
        [parsed_ftp.netloc,parsed_ftp.path], timeout=TIMEOUT,
        basename=True, pattern=R1, sort=True)

    #-- sync each data file
    out = []
    #-- set multiprocessing start method
    ctx = multiprocessing.get_context("fork")
    #-- sync in parallel with multiprocessing Pool
    pool = ctx.Pool(processes=PROCESSES)
    #-- download remote MAR files to local directory
    for colname,collastmod in zip(remote_files,remote_times):
        remote_path = [parsed_ftp.netloc, parsed_ftp.path, colname]
        local_file = os.path.join(DIRECTORY,colname)
        kwds = dict(TIMEOUT=TIMEOUT, CLOBBER=CLOBBER, MODE=MODE)
        out.append(pool.apply_async(multiprocess_sync,
            args=(remote_path,collastmod,local_file),
            kwds=kwds))
    #-- start multiprocessing jobs
    #-- close the pool
    #-- prevents more tasks from being submitted to the pool
    pool.close()
    #-- exit the completed processes
    pool.join()
    #-- print the output string
    for output in out:
        temp = output.get()
        logging.info(temp) if temp else None

    #-- close log file and set permissions level to MODE
    if LOG:
        os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

#-- PURPOSE: wrapper for running the sync program in multiprocessing mode
def multiprocess_sync(*args, **kwds):
    try:
        output = ftp_mirror_file(*args, **kwds)
    except Exception as e:
        #-- if there has been an error exception
        #-- print the type, value, and stack trace of the
        #-- current exception being handled
        logging.critical('process id {0:d} failed'.format(os.getpid()))
        logging.error(traceback.format_exc())
    else:
        return output

#-- PURPOSE: pull file from a remote host checking if file exists locally
#-- and if the remote file is newer than the local file
def ftp_mirror_file(remote_path, remote_mtime, local_file, **kwargs):
    #-- check if local version of file exists
    local_hash = SMBcorr.utilities.get_hash(local_file)
    if os.access(local_file, os.F_OK):
        #-- check last modification time of local file
        local_mtime = os.stat(local_file).st_mtime
        #-- if remote file is newer: overwrite the local file
        if (remote_mtime > local_mtime):
            TEST = True
            OVERWRITE = ' (overwrite)'
    else:
        TEST = True
        OVERWRITE = ' (new)'
    #-- if file does not exist locally, is to be overwritten, or CLOBBER is set
    if TEST or kwargs['CLOBBER']:
        #-- output string for printing files transferred
        output = '{0} -->\n\t{1}{2}\n'.format(posixpath.join(*remote_path),
            local_file,OVERWRITE)
        #-- copy remote file contents to local file
        SMBcorr.utilities.from_ftp(remote_path,timeout=kwargs['TIMEOUT'],
            local=local_file,hash=local_hash)
        #-- keep remote modification time of file and local access time
        os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
        os.chmod(local_file, kwargs['MODE'])
        #-- return the output string
        return output

#-- This is the main part of the program that calls the individual modules
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Syncs MAR regional climate outputs for a given ftp url
            """
    )
    parser.add_argument('url',
        type=str,
        help='MAR ftp url')
    #-- working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- years of data to sync
    parser.add_argument('--year','-Y',
        type=int, nargs='+',
        help='Years to sync')
    #-- Output log file
    parser.add_argument('--log','-l',
        default=False, action='store_true',
        help='Output log file')
    #-- run sync in series if processes is 0
    parser.add_argument('--np','-P',
        metavar='PROCESSES', type=int, default=1,
        help='Number of processes to use in file downloads')
    #-- connection timeout
    parser.add_argument('--timeout','-T',
        type=int, default=120,
        help='Timeout in seconds for blocking operations')
    #-- clobber will overwrite the existing data
    parser.add_argument('--clobber','-C',
        default=False, action='store_true',
        help='Overwrite existing data')
    #-- permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    args,_ = parser.parse_known_args()

    #-- check internet connection
    if SMBcorr.utilities.check_ftp_connection('ftp.climato.be'):
        #-- run program for parsed ftp
        parsed_ftp = SMBcorr.utilities.urlparse.urlparse(args.url)
        ftp_mar_data(parsed_ftp,
            DIRECTORY=args.directory,
            YEARS=args.year,
            TIMEOUT=args.timeout,
            LOG=args.log,
            PROCESSES=args.np,
            CLOBBER=args.clobber,
            MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
