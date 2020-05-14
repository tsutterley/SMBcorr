#!/usr/bin/env python
u"""
gesdisc_merra_download.py
Written by Tyler Sutterley (09/2019)

This program downloads MERRA-2 products using a links list provided by the
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
    -D X, --directory: Full path to output directory
    --user: username for NASA Earthdata Login
    -M X, --mode=X: Local permissions mode of the directories and files created
    --log: output log of files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        (http://python-future.org/)

UPDATE HISTORY:
    Updated 09/2019: added ssl context to urlopen headers
    Updated 08/2019: new GESDISC server and links list file format
        increased timeout to 20 seconds
    Updated 06/2018: using python3 compatible octal, input and urllib
    Written 03/2018
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
def gesdisc_merra_download(links_list_file, USER='', PASSWORD='',
    DIRECTORY=None, LOG=False, MODE=None):
    #-- check if DIRECTORY exists and recursively create if not
    os.makedirs(DIRECTORY,MODE) if not os.path.exists(DIRECTORY) else None

    #-- create log file with list of synchronized files (or print to terminal)
    if LOG:
        #-- format: NASA_GESDISC_MERRA2_download_2002-04-01.log
        today = time.strftime('%Y-%m-%d',time.localtime())
        LOGFILE = 'NASA_GESDISC_MERRA2_download_{0}.log'.format(today)
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

    #-- read the links list file
    with open(links_list_file,'rb') as fileID:
        lines = fileID.read().decode("utf-8-sig").encode("utf-8").splitlines()

    #-- for each line in the links_list_file
    for f in lines:
        #-- extract filename from url
        if re.search(b'LABEL\=(.*?)\&SHORTNAME',f):
            FILE, = re.findall('LABEL\=(.*?)\&SHORTNAME', f.decode('utf-8'))
        elif re.search(b'MERRA2_(\d+).(.*?).(\d+).(.*?).nc',f):
            rx = re.compile('MERRA2_(\d+)\.(.*?)\.(\d+)\.(.*?).nc')
            MOD,DSET,YMD,AUX = rx.findall(f.decode('utf-8')).pop()
            FILE = 'MERRA2_{0}.{1}.{2}.SUB.nc'.format(MOD,DSET,YMD)
        else:
            FILE = posixpath.split(f.decode('utf-8'))[1]
        #-- Printing files transferred
        local_file = os.path.join(DIRECTORY,FILE)
        print('{0} -->\n\t{1}\n'.format(f.decode('utf-8'),local_file), file=fid)
        #-- Create and submit request. There are a wide range of exceptions
        #-- that can be thrown here, including HTTPError and URLError.
        request = urllib2.Request(url=f.decode('utf-8').strip())
        response = urllib2.urlopen(request, timeout=20)
        #-- chunked transfer encoding size
        CHUNK = 16 * 1024
        #-- copy contents to local file using chunked transfer encoding
        #-- transfer should work properly with ascii and binary data formats
        with open(local_file, 'wb') as f:
            shutil.copyfileobj(response, f, CHUNK)
        #-- change permissions to MODE
        os.chmod(local_file, MODE)
        #-- close request
        request = None

    #-- close log file and set permissions level to MODE
    if LOG:
        fid.close()
        os.chmod(os.path.join(DIRECTORY,LOGFILE), MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\t\tFull path to working data directory')
    print(' -U X, --user=X\t\tUsername for NASA Earthdata Login')
    print(' -M X, --mode=X\t\tPermission mode of directories and files synced')
    print(' -l, --log\t\tOutput log file')
    today = time.strftime('%Y-%m-%d',time.localtime())
    LOGFILE = 'NASA_GESDISC_MERRA2_download_{0}.log'.format(today)
    print('    Log file format: {}\n'.format(LOGFILE))

#-- Main program that calls gesdisc_merra_download()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','user=','log','mode=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'hD:U:M:l',long_options)

    #-- command line parameters
    DIRECTORY = os.getcwd()
    USER = ''
    LOG = False
    #-- permissions mode of the local directories and files (number in octal)
    MODE = 0o775
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("--directory"):
            DIRECTORY = os.path.expanduser(arg)
        elif opt in ("-U","--user"):
            USER = arg
        elif opt in ("-l","--log"):
            LOG = True
        elif opt in ("-M","--mode"):
            MODE = int(arg, 8)

    #-- NASA Earthdata hostname
    HOST = 'urs.earthdata.nasa.gov'
    #-- check that NASA Earthdata credentials were entered
    if not USER:
        USER = builtins.input('Username for {0}: '.format(HOST))
    #-- enter password securely from command-line
    PASSWORD = getpass.getpass('Password for {0}@{1}: '.format(USER,HOST))

    #-- check internet connection before attempting to run program
    if check_connection():
        #-- for each links list file from GESDISC
        for fi in arglist:
            gesdisc_merra_download(os.path.expanduser(fi), USER=USER,
                PASSWORD=PASSWORD, DIRECTORY=DIRECTORY, LOG=LOG, MODE=MODE)

#-- run main program
if __name__ == '__main__':
    main()
