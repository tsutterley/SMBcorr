#!/usr/bin/env python
u"""
ftp_mar_data.py
Written by Tyler Sutterley (05/2020)

Syncs MAR regional climate outputs for a given ftp url
    ftp://ftp.climato.be/fettweis

CALLING SEQUENCE:
    python ftp_mar_data.py --directory=<path> <ftp://url>

INPUTS:
    full ftp url

COMMAND LINE OPTIONS:
    --help: list the command line options
    -P X, --np=X: Run in parallel with X number of processes
    -D X, --directory=X: full path to working data directory
    -Y X, --year=X: Reduce files to years of interest
    -C, --clobber: Overwrite existing data in transfer
    -M X, --mode=X: Local permissions mode of the directories and files synced

UPDATE HISTORY:
    Updated 05/2020: added years option to reduce list of files
    Updated 11/2019: added multiprocessing option to run in parallel
    Written 07/2019
"""
from __future__ import print_function

import sys
import getopt
import os
import re
import calendar, time
import ftplib
import traceback
import posixpath
import multiprocessing
if sys.version_info[0] == 2:
    import urlparse
else:
    import urllib.parse as urlparse

#-- PURPOSE: check internet connection
def check_connection():
    #-- attempt to connect to ftp host for MAR datasets
    try:
        f = ftplib.FTP('ftp.climato.be')
        f.login()
        f.voidcmd("NOOP")
    except IOError:
        raise RuntimeError('Check internet connection')
    else:
        return True

#-- PURPOSE: sync local MAR files with ftp and handle error exceptions
def ftp_mar_data(netloc, remote_file, local_file, CLOBBER=False, MODE=0o775):
    #-- try to download the file
    try:
        #-- connect and login to MAR ftp server
        ftp = ftplib.FTP(netloc)
        ftp.login()
        ftp_mirror_file(ftp,remote_file,local_file,CLOBBER=CLOBBER,MODE=MODE)
    except:
        #-- if there has been an error exception
        #-- print the type, value, and stack trace of the
        #-- current exception being handled
        print('process id {0:d} failed'.format(os.getpid()))
        traceback.print_exc()
    else:
        #-- close the ftp connection
        ftp.quit()

#-- PURPOSE: pull file from a remote host checking if file exists locally
#-- and if the remote file is newer than the local file
def ftp_mirror_file(ftp, remote_file, local_file, CLOBBER=False, MODE=0o775):
    #-- if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
    #-- get last modified date of remote file and convert into unix time
    mdtm = ftp.sendcmd('MDTM {0}'.format(remote_file))
    remote_mtime = calendar.timegm(time.strptime(mdtm[4:],"%Y%m%d%H%M%S"))
    #-- check if local version of file exists
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
    if TEST or CLOBBER:
        #-- Printing files transferred
        print('{0} -->'.format(posixpath.join('ftp://',ftp.host,remote_file)))
        print('\t{0}{1}\n'.format(local_file,OVERWRITE))
        #-- copy remote file contents to local file
        with open(local_file, 'wb') as f:
            ftp.retrbinary('RETR {0}'.format(remote_file), f.write)
        #-- keep remote modification time of file and local access time
        os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
        os.chmod(local_file, MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -P X, --np=X\tRun in parallel with X number of processes')
    print(' -D X, --directory=X\tWorking Data Directory')
    print(' -Y X, --year=X\tReduce files to years of interest')
    print(' -C, --clobber\t\tOverwrite existing data in transfer')
    print(' -M X, --mode=X\t\tPermission mode of directories and files\n')

#-- This is the main part of the program that calls the individual modules
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','np=','directory=','year=','clobber','mode=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'hP:D:Y:CM:',long_options)

    #-- command line parameters
    local_dir = os.getcwd()
    #-- years to sync (default all)
    YEARS = '\d+'
    #-- number of processes
    PROCESSES = 1
    CLOBBER = False
    #-- permissions mode of the local directories and files (number in octal)
    MODE = 0o775
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("-D","--directory"):
            local_dir = os.path.expanduser(arg)
        elif opt in ("-Y","--year"):
            YEARS = '|'.join(arg.split(','))
        elif opt in ("-P","--np"):
            PROCESSES = int(arg)
        elif opt in ("-C","--clobber"):
            CLOBBER = True
        elif opt in ("-M","--mode"):
            MODE = int(arg, 8)

    #-- need to input a ftp path
    if not arglist:
        raise IOError('Need to input a path to the MAR ftp server')

    #-- check internet connection
    if check_connection():
        #-- check if local directory exists and recursively create if not
        os.makedirs(local_dir,MODE) if not os.path.exists(local_dir) else None

        #-- connect and login to MAR ftp server
        #-- get list of files to download
        parsed_ftp = urlparse.urlparse(arglist[0])
        ftp = ftplib.FTP(parsed_ftp.netloc)
        ftp.login()
        # find files and reduce to years of interest if specified
        remote_files = sorted([f for f in ftp.nlst(parsed_ftp.path)
            if re.search(YEARS,posixpath.basename(f))])
        ftp.quit()

        #-- run in parallel with multiprocessing Pool
        pool = multiprocessing.Pool(processes=PROCESSES)
        #-- download remote MAR files to local directory
        for j,remote_file in enumerate(remote_files):
            #-- extract filename
            url,fi = posixpath.split(remote_file)
            args = (parsed_ftp.netloc, remote_file, os.path.join(local_dir,fi),)
            kwds = dict(CLOBBER=CLOBBER, MODE=MODE)
            pool.apply_async(ftp_mar_data, args=args, kwds=kwds)
        #-- start multiprocessing jobs
        #-- close the pool
        #-- prevents more tasks from being submitted to the pool
        pool.close()
        #-- exit the completed processes
        pool.join()

#-- run main program
if __name__ == '__main__':
    main()
