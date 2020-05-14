#!/usr/bin/env python
u"""
gesdisc_merra_sync.py
Written by Tyler Sutterley (09/2019)

This program syncs MERRA-2 surface mass balance (SMB) related products from the
    Goddard Earth Sciences Data and Information Server Center
    https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python

Register with NASA Earthdata Login system:
    https://urs.earthdata.nasa.gov

Add "NASA GESDISC DATA ARCHIVE" to Earthdata Applications:
    https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ

CALLING SEQUENCE:
    python gesdisc_merra_sync.py --user=<username>
    where <username> is your NASA Earthdata username

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory: Working data directory (default: $PYTHONDATA)
    -Y X, --year=X: years to sync separated by commas
    --user: username for NASA Earthdata Login
    -M X, --mode=X: Local permissions mode of the directories and files synced
    --log: output log of files downloaded
    --list: print files to be transferred, but do not execute transfer
    --clobber: Overwrite existing data in transfer

PYTHON DEPENDENCIES:
    lxml: Pythonic XML and HTML processing library using libxml2/libxslt
        http://lxml.de/
        https://github.com/lxml/lxml
    future: Compatibility layer between Python 2 and Python 3
        (http://python-future.org/)

UPDATE HISTORY:
    Updated 09/2019: added ssl context to urlopen headers
    Updated 06/2018: using python3 compatible octal, input and urllib
    Updated 03/2018: --directory sets base directory similar to other programs
    Updated 08/2017: use raw_input() to enter NASA Earthdata credentials rather
        than exiting with error
    Updated 05/2017: exception if NASA Earthdata credentials weren't entered
        using os.makedirs to recursively create directories
        using getpass to enter server password securely (remove --password)
    Updated 04/2017: using lxml to parse HTML for files and modification dates
        minor changes to check_connection function to parallel other programs
    Written 11/2016
"""
from __future__ import print_function
import future.standard_library

import sys
import os
import re
import ssl
import getopt
import shutil
import getpass
import builtins
import posixpath
import lxml.etree
import calendar, time
if sys.version_info[0] == 2:
    from cookielib import CookieJar
    import urllib2
else:
    from http.cookiejar import CookieJar
    import urllib.request as urllib2

#-- PURPOSE: check internet connection
def check_connection():
    #-- attempt to connect to http GESDISC host
    try:
        HOST = 'http://disc.sci.gsfc.nasa.gov/'
        urllib2.urlopen(HOST,timeout=20,context=ssl.SSLContext())
    except urllib2.URLError:
        raise RuntimeError('Check internet connection')
    else:
        return True

#-- PURPOSE: sync local MERRA-2 files with GESDISC server
def gesdisc_merra_sync(DIRECTORY, YEARS, USER='', PASSWORD='',
    LOG=False, LIST=False, MODE=None, CLOBBER=False):
    #-- recursively create directory if non-existent
    os.makedirs(DIRECTORY,MODE) if not os.path.exists(DIRECTORY) else None
    
    #-- create log file with list of synchronized files (or print to terminal)
    if LOG:
        #-- format: NASA_GESDISC_MERRA2_sync_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = 'NASA_GESDISC_MERRA2_sync_{0}.log'.format(today)
        fid = open(os.path.join(DIRECTORY,LOGFILE),'w')
        print('NASA MERRA-2 Sync Log ({0})'.format(today), file=fid)
    else:
        #-- standard output (terminal output)
        fid = sys.stdout

    #-- https://docs.python.org/3/howto/urllib2.html#id5
    #-- create a password manager
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    #-- Add the username and password for NASA Earthdata Login system
    password_mgr.add_password(None, 'https://urs.earthdata.nasa.gov',
        USER, PASSWORD)
    #-- compile HTML parser for lxml
    parser = lxml.etree.HTMLParser()
    #-- Create cookie jar for storing cookies. This is used to store and return
    #-- the session cookie given to use by the data server (otherwise will just
    #-- keep sending us back to Earthdata Login to authenticate).
    cookie_jar = CookieJar()
    #-- create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(
        urllib2.HTTPBasicAuthHandler(password_mgr),
        urllib2.HTTPSHandler(context=ssl.SSLContext()),
        urllib2.HTTPCookieProcessor(cookie_jar))
    #-- Now all calls to urllib2.urlopen use our opener.
    urllib2.install_opener(opener)
    #-- All calls to urllib2.urlopen will now use handler
    #-- Make sure not to include the protocol in with the URL, or
    #-- HTTPPasswordMgrWithDefaultRealm will be confused.

    #-- MERRA-2 data remote base directory
    HOST = posixpath.join('http://goldsmr4.gesdisc.eosdis.nasa.gov','data',
        'MERRA2_MONTHLY')

    #-- compile regular expression operator for years to sync
    regex_pattern = '|'.join('{0:d}'.format(y) for y in YEARS)
    R1 = re.compile('({0})'.format(regex_pattern), re.VERBOSE)
    #-- compile regular expression operator to find MERRA2 files
    R2 = re.compile('MERRA2_(.*?).nc4(.xml)?', re.VERBOSE)

    #-- for each MERRA-2 product to sync
    for PRODUCT in ['M2TMNXINT.5.12.4','M2TMNXGLC.5.12.4']:
        print('PRODUCT={0}'.format(PRODUCT), file=fid)
        #-- open connection with GESDISC server at remote directory
        req = urllib2.Request(url=posixpath.join(HOST,PRODUCT))
        #-- read and parse request for subdirectories (find column names)
        tree = lxml.etree.parse(urllib2.urlopen(req), parser)
        colnames = tree.xpath('//tr/td[not(@*)]//a/@href')
        #-- find remote yearly directories for PRODUCT
        remote_sub = [sd for sd in colnames if R1.match(sd)]
        for Y in remote_sub:
            #-- check if local directory exists and recursively create if not
            if (not os.access(os.path.join(DIRECTORY,PRODUCT,Y), os.F_OK)):
                os.makedirs(os.path.join(DIRECTORY,PRODUCT,Y), MODE)
            #-- open connection with GESDISC server at remote directory
            req = urllib2.Request(url=posixpath.join(HOST,PRODUCT,Y))
            #-- read and parse request for files (find names and modified dates)
            tree = lxml.etree.parse(urllib2.urlopen(req), parser)
            colnames = tree.xpath('//tr/td[not(@*)]//a/@href')
            collastmod = tree.xpath('//tr/td[@align="right"][1]/text()')
            #-- find remote files for PRODUCT and YEAR
            remote_file_lines=[i for i,f in enumerate(colnames) if R2.match(f)]
            for i in remote_file_lines:
                #-- local and remote versions of the file
                FILE = colnames[i]
                local_file = os.path.join(DIRECTORY,PRODUCT,Y,FILE)
                remote_file = posixpath.join(HOST,PRODUCT,Y,FILE)
                #-- get last modified date of file and convert into unix time
                file_date=time.strptime(collastmod[i].rstrip(),'%d-%b-%Y %H:%M')
                remote_mtime = calendar.timegm(file_date)
                #-- copy file from remote directory checking modification times
                http_pull_file(fid, remote_file, remote_mtime, local_file,
                    LIST, CLOBBER, MODE)
            #-- close request
            req = None

    #-- close log file and set permissions level to MODE
    if LOG:
        fid.close()
        os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

#-- PURPOSE: pull file from a remote host checking if file exists locally
#-- and if the remote file is newer than the local file
def http_pull_file(fid,remote_file,remote_mtime,local_file,LIST,CLOBBER,MODE):
    #-- if file exists in file system: check if remote file is newer
    TEST = False
    OVERWRITE = ' (clobber)'
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
        print('{0} --> '.format(remote_file), file=fid)
        print('\t{0}{1}\n'.format(local_file,OVERWRITE), file=fid)
        #-- if executing copy command (not only printing the files)
        if not LIST:
            #-- Create and submit request. There are a wide range of exceptions
            #-- that can be thrown here, including HTTPError and URLError.
            request = urllib2.Request(remote_file)
            response = urllib2.urlopen(request)
            #-- chunked transfer encoding size
            CHUNK = 16 * 1024
            #-- copy contents to local file using chunked transfer encoding
            #-- transfer should work properly with ascii and binary data formats
            with open(local_file, 'wb') as f:
                shutil.copyfileobj(response, f, CHUNK)
            #-- keep remote modification time of file and local access time
            os.utime(local_file, (os.stat(local_file).st_atime, remote_mtime))
            os.chmod(local_file, MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\t\tWorking data directory')
    print(' -Y X, --year=X\t\tYears to sync separated by commas')
    print(' -U X, --user=X\t\tUsername for NASA Earthdata Login')
    print(' -M X, --mode=X\t\tPermission mode of directories and files synced')
    print(' -L, --list\t\tOnly print files that are to be transferred')
    print(' -C, --clobber\t\tOverwrite existing data in transfer')
    print(' -l, --log\t\tOutput log file')
    today = time.strftime('%Y-%m-%d',time.localtime())
    LOGFILE = 'NASA_GESDISC_MERRA2_sync_{0}.log'.format(today)
    print('    Log file format: {}\n'.format(LOGFILE))

#-- Main program that calls gesdisc_merra_sync()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','year=','user=','list','log',
        'mode=','clobber']
    optlist,arglist = getopt.getopt(sys.argv[1:],'hD:Y:U:LCM:l',long_options)

    #-- command line parameters
    DIRECTORY = os.getcwd()
    years = [y for y in range(1980,2020)]
    USER = ''
    LIST = False
    LOG = False
    #-- permissions mode of the local directories and files (number in octal)
    MODE = 0o775
    CLOBBER = False
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("--directory"):
            DIRECTORY = os.path.expanduser(arg)
        elif opt in ("-Y","--year"):
            years = [int(Y) for Y in arg.split(',')]
        elif opt in ("-U","--user"):
            USER = arg
        elif opt in ("-L","--list"):
            LIST = True
        elif opt in ("-l","--log"):
            LOG = True
        elif opt in ("-M","--mode"):
            MODE = int(arg, 8)
        elif opt in ("-C","--clobber"):
            CLOBBER = True

    #-- NASA Earthdata hostname
    HOST = 'urs.earthdata.nasa.gov'
    #-- check that NASA Earthdata credentials were entered
    if not USER:
        USER = builtins.input('Username for {0}: '.format(HOST))
    #-- enter password securely from command-line
    PASSWORD = getpass.getpass('Password for {0}@{1}: '.format(USER,HOST))

    #-- check internet connection before attempting to run program
    if check_connection():
        gesdisc_merra_sync(DIRECTORY, years, USER=USER, PASSWORD=PASSWORD,
            LOG=LOG, LIST=LIST, MODE=MODE, CLOBBER=CLOBBER)

#-- run main program
if __name__ == '__main__':
    main()
