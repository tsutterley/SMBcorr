#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (06/2022)
Download and management utilities for syncing time and auxiliary files

PYTHON DEPENDENCIES:
    lxml: processing XML and HTML in Python
        https://pypi.python.org/pypi/lxml

UPDATE HISTORY:
    Updated 06/2022: add NASA Common Metadata Repository (CMR) queries
        added function to build GES DISC subsetting API requests
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 10/2021: build python logging instance for handling verbose output
    Updated 09/2021: added generic list from Apache http server
    Updated 07/2021: add parser for converting file files to arguments
    Updated 03/2021: added sha1 option for retrieving file hashes
    Updated 01/2021: added username and password to ftp functions
        added ftp connection check
    Updated 12/2020: added file object keyword for downloads if verbose
        add url split function for creating url location lists
    Updated 11/2020: normalize source and destination paths in copy
        make context an optional keyword argument in from_http
    Updated 09/2020: copy from http and https to bytesIO object in chunks
    Written 08/2020
"""
from __future__ import print_function, division

import sys
import os
import re
import io
import ssl
import json
import netrc
import ftplib
import shutil
import base64
import socket
import inspect
import hashlib
import logging
import posixpath
import lxml.etree
import calendar, time
import dateutil.parser
if sys.version_info[0] == 2:
    from cookielib import CookieJar
    from urllib import urlencode
    import urllib2
    import urlparse
else:
    from http.cookiejar import CookieJar
    from urllib.parse import urlencode
    import urllib.request as urllib2
    import urllib.parse as urlparse

#-- PURPOSE: get absolute path within a package from a relative path
def get_data_path(relpath):
    """
    Get the absolute path within a package from a relative path

    Parameters
    ----------
    relpath: str,
        relative path
    """
    #-- current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    if isinstance(relpath,list):
        #-- use *splat operator to extract from list
        return os.path.join(filepath,*relpath)
    elif isinstance(relpath,str):
        return os.path.join(filepath,relpath)

#-- PURPOSE: get the hash value of a file
def get_hash(local, algorithm='MD5'):
    """
    Get the hash value from a local file or BytesIO object

    Parameters
    ----------
    local: obj or str
        BytesIO object or path to file
    algorithm: str, default 'MD5'
        hashing algorithm for checksum validation

            - ``'MD5'``: Message Digest
            - ``'sha1'``: Secure Hash Algorithm
    """
    #-- check if open file object or if local file exists
    if isinstance(local, io.IOBase):
        if (algorithm == 'MD5'):
            return hashlib.md5(local.getvalue()).hexdigest()
        elif (algorithm == 'sha1'):
            return hashlib.sha1(local.getvalue()).hexdigest()
    elif os.access(os.path.expanduser(local),os.F_OK):
        #-- generate checksum hash for local file
        #-- open the local_file in binary read mode
        with open(os.path.expanduser(local), 'rb') as local_buffer:
            #-- generate checksum hash for a given type
            if (algorithm == 'MD5'):
                return hashlib.md5(local_buffer.read()).hexdigest()
            elif (algorithm == 'sha1'):
                return hashlib.sha1(local_buffer.read()).hexdigest()
    else:
        return ''

#-- PURPOSE: recursively split a url path
def url_split(s):
    """
    Recursively split a url path into a list

    Parameters
    ----------
    s: str
        url string
    """
    head, tail = posixpath.split(s)
    if head in ('http:','https:','ftp:','s3:'):
        return s,
    elif head in ('', posixpath.sep):
        return tail,
    return url_split(head) + (tail,)

#-- PURPOSE: convert file lines to arguments
def convert_arg_line_to_args(arg_line):
    """
    Convert file lines to arguments

    Parameters
    ----------
    arg_line: str
        line string containing a single argument and/or comments
    """
    #-- remove commented lines and after argument comments
    for arg in re.sub(r'\#(.*?)$',r'',arg_line).split():
        if not arg.strip():
            continue
        yield arg

#-- PURPOSE: returns the Unix timestamp value for a formatted date string
def get_unix_time(time_string, format='%Y-%m-%d %H:%M:%S'):
    """
    Get the Unix timestamp value for a formatted date string

    Parameters
    ----------
    time_string: str
        formatted time string to parse
    format: str, default '%Y-%m-%d %H:%M:%S'
        format for input time string
    """
    try:
        parsed_time = time.strptime(time_string.rstrip(), format)
    except (TypeError, ValueError):
        pass
    else:
        return calendar.timegm(parsed_time)
    #-- try parsing with dateutil
    try:
        parsed_time = dateutil.parser.parse(time_string.rstrip())
    except (TypeError, ValueError):
        return None
    else:
        return parsed_time.timestamp()

#-- PURPOSE: output a time string in isoformat
def isoformat(time_string):
    """
    Reformat a date string to ISO formatting

    Parameters
    ----------
    time_string: str
        formatted time string to parse
    """
    #-- try parsing with dateutil
    try:
        parsed_time = dateutil.parser.parse(time_string.rstrip())
    except (TypeError, ValueError):
        return None
    else:
        return parsed_time.isoformat()

#-- PURPOSE: rounds a number to an even number less than or equal to original
def even(value):
    """
    Rounds a number to an even number less than or equal to original

    Parameters
    ----------
    value: float
        number to be rounded
    """
    return 2*int(value//2)

#-- PURPOSE: rounds a number upward to its nearest integer
def ceil(value):
    """
    Rounds a number upward to its nearest integer

    Parameters
    ----------
    value: float
        number to be rounded upward
    """
    return -int(-value//1)

#-- PURPOSE: make a copy of a file with all system information
def copy(source, destination, move=False, **kwargs):
    """
    Copy or move a file with all system information

    Parameters
    ----------
    source: str
        source file
    destination: str
        copied destination file
    move: bool, default False
        remove the source file
    """
    source = os.path.abspath(os.path.expanduser(source))
    destination = os.path.abspath(os.path.expanduser(destination))
    #-- log source and destination
    logging.info('{0} -->\n\t{1}'.format(source,destination))
    shutil.copyfile(source, destination)
    shutil.copystat(source, destination)
    if move:
        os.remove(source)

#-- PURPOSE: check ftp connection
def check_ftp_connection(HOST, username=None, password=None):
    """
    Check internet connection with ftp host

    Parameters
    ----------
    HOST: str
        remote ftp host
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    """
    #-- attempt to connect to ftp host
    try:
        f = ftplib.FTP(HOST)
        f.login(username, password)
        f.voidcmd("NOOP")
    except IOError:
        raise RuntimeError('Check internet connection')
    except ftplib.error_perm:
        raise RuntimeError('Check login credentials')
    else:
        return True

#-- PURPOSE: list a directory on a ftp host
def ftp_list(HOST, username=None, password=None, timeout=None,
    basename=False, pattern=None, sort=False):
    """
    List a directory on a ftp host

    Parameters
    ----------
    HOST: str or list
        remote ftp host path split as list
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    basename: bool, default False
        return the file or directory basename instead of the full path
    pattern: str or NoneType, default None
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    output: list
        items in a directory
    mtimes: list
        last modification times for items in the directory
    """
    #-- verify inputs for remote ftp host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    #-- try to connect to ftp host
    try:
        ftp = ftplib.FTP(HOST[0],timeout=timeout)
    except (socket.gaierror,IOError):
        raise RuntimeError('Unable to connect to {0}'.format(HOST[0]))
    else:
        ftp.login(username,password)
        #-- list remote path
        output = ftp.nlst(posixpath.join(*HOST[1:]))
        #-- get last modified date of ftp files and convert into unix time
        mtimes = [None]*len(output)
        #-- iterate over each file in the list and get the modification time
        for i,f in enumerate(output):
            try:
                #-- try sending modification time command
                mdtm = ftp.sendcmd('MDTM {0}'.format(f))
            except ftplib.error_perm:
                #-- directories will return with an error
                pass
            else:
                #-- convert the modification time into unix time
                mtimes[i] = get_unix_time(mdtm[4:], format="%Y%m%d%H%M%S")
        #-- reduce to basenames
        if basename:
            output = [posixpath.basename(i) for i in output]
        #-- reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(output) if re.search(pattern,f)]
            #-- reduce list of listed items and last modified times
            output = [output[indice] for indice in i]
            mtimes = [mtimes[indice] for indice in i]
        #-- sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(output), key=lambda i: i[1])]
            #-- sort list of listed items and last modified times
            output = [output[indice] for indice in i]
            mtimes = [mtimes[indice] for indice in i]
        #-- close the ftp connection
        ftp.close()
        #-- return the list of items and last modified times
        return (output, mtimes)

#-- PURPOSE: download a file from a ftp host
def from_ftp(HOST, username=None, password=None, timeout=None,
    local=None, hash='', chunk=8192, verbose=False, fid=sys.stdout,
    mode=0o775):
    """
    Download a file from a ftp host

    Parameters
    ----------
    HOST: str or list
        remote ftp host path
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    local: str or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 8192
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    fid: obj, default sys.stdout
        open file object to print if verbose
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    #-- create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    #-- verify inputs for remote ftp host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    #-- try downloading from ftp
    try:
        #-- try to connect to ftp host
        ftp = ftplib.FTP(HOST[0], timeout=timeout)
    except (socket.gaierror,IOError):
        raise RuntimeError('Unable to connect to {0}'.format(HOST[0]))
    else:
        ftp.login(username,password)
        #-- remote path
        ftp_remote_path = posixpath.join(*HOST[1:])
        #-- copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        ftp.retrbinary('RETR {0}'.format(ftp_remote_path),
            remote_buffer.write, blocksize=chunk)
        remote_buffer.seek(0)
        #-- save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        #-- generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        #-- get last modified date of remote file and convert into unix time
        mdtm = ftp.sendcmd('MDTM {0}'.format(ftp_remote_path))
        remote_mtime = get_unix_time(mdtm[4:], format="%Y%m%d%H%M%S")
        #-- compare checksums
        if local and (hash != remote_hash):
            #-- convert to absolute path
            local = os.path.abspath(local)
            #-- create directory if non-existent
            if not os.access(os.path.dirname(local), os.F_OK):
                os.makedirs(os.path.dirname(local), mode)
            #-- print file information
            args = (posixpath.join(*HOST),local)
            logging.info('{0} -->\n\t{1}'.format(*args))
            #-- store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with open(os.path.expanduser(local), 'wb') as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            #-- change the permissions mode
            os.chmod(local,mode)
            #-- keep remote modification time of file and local access time
            os.utime(local, (os.stat(local).st_atime, remote_mtime))
        #-- close the ftp connection
        ftp.close()
        #-- return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer

#-- PURPOSE: check internet connection
def check_connection(HOST):
    """
    Check internet connection with http host

    Parameters
    ----------
    HOST: str
        remote http host
    """
    #-- attempt to connect to http host
    try:
        urllib2.urlopen(HOST, timeout=20, context=ssl.SSLContext())
    except urllib2.URLError:
        raise RuntimeError('Check internet connection')
    else:
        return True

#-- PURPOSE: list a directory on an Apache http Server
def http_list(HOST, timeout=None, context=ssl.SSLContext(),
    parser=lxml.etree.HTMLParser(), format='%Y-%m-%d %H:%M',
    pattern='', sort=False):
    """
    List a directory on an Apache http Server

    Parameters
    ----------
    HOST: str or list
        remote http host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default ssl.SSLContext()
        SSL context for url opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for lxml
    format: str, default '%Y-%m-%d %H:%M'
        format for input time string
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    """
    #-- verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    #-- try listing from http
    try:
        #-- Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except (urllib2.HTTPError, urllib2.URLError):
        raise Exception('List error from {0}'.format(posixpath.join(*HOST)))
    else:
        #-- read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response, parser)
        colnames = tree.xpath('//tr/td[not(@*)]//a/@href')
        #-- get the Unix timestamp value for a modification time
        collastmod = [get_unix_time(i,format=format)
            for i in tree.xpath('//tr/td[@align="right"][1]/text()')]
        #-- reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(colnames) if re.search(pattern, f)]
            #-- reduce list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        #-- sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(colnames), key=lambda i: i[1])]
            #-- sort list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        #-- return the list of column names and last modified times
        return (colnames, collastmod)

#-- PURPOSE: download a file from a http host
def from_http(HOST, timeout=None, context=ssl.SSLContext(),
    local=None, hash='', chunk=16384, verbose=False, fid=sys.stdout,
    mode=0o775):
    """
    Download a file from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default ssl.SSLContext()
        SSL context for url opener object
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    local: str or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    fid: obj, default sys.stdout
        open file object to print if verbose
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    #-- create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    #-- verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    #-- try downloading from http
    try:
        #-- Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except:
        raise Exception('Download error from {0}'.format(posixpath.join(*HOST)))
    else:
        #-- copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        #-- save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        #-- generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        #-- compare checksums
        if local and (hash != remote_hash):
            #-- convert to absolute path
            local = os.path.abspath(local)
            #-- create directory if non-existent
            if not os.access(os.path.dirname(local), os.F_OK):
                os.makedirs(os.path.dirname(local), mode)
            #-- print file information
            args = (posixpath.join(*HOST),local)
            logging.info('{0} -->\n\t{1}'.format(*args))
            #-- store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with open(os.path.expanduser(local), 'wb') as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            #-- change the permissions mode
            os.chmod(local,mode)
        #-- return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer

#-- PURPOSE: "login" to NASA Earthdata with supplied credentials
def build_opener(username, password, context=ssl.SSLContext(),
    password_manager=False, get_ca_certs=False, redirect=False,
    authorization_header=True, urs='https://urs.earthdata.nasa.gov'):
    """
    build urllib opener for NASA Earthdata with supplied credentials

    Parameters
    ----------
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    context: obj, default ssl.SSLContext()
        SSL context for url opener object
    password_manager: bool, default False
        Create password manager context using default realm
    get_ca_certs: bool, default False
        Get list of loaded “certification authority” certificates
    redirect: bool, default False
        Create redirect handler object
    authorization_header: bool, default True
        Add base64 encoded authorization header to opener
    urs: str, default 'https://urs.earthdata.nasa.gov'
        Earthdata login URS 3 host
    """
    #-- https://docs.python.org/3/howto/urllib2.html#id5
    handler = []
    #-- create a password manager
    if password_manager:
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        #-- Add the username and password for NASA Earthdata Login system
        password_mgr.add_password(None, urs, username, password)
        handler.append(urllib2.HTTPBasicAuthHandler(password_mgr))
    #-- Create cookie jar for storing cookies. This is used to store and return
    #-- the session cookie given to use by the data server (otherwise will just
    #-- keep sending us back to Earthdata Login to authenticate).
    cookie_jar = CookieJar()
    handler.append(urllib2.HTTPCookieProcessor(cookie_jar))
    #-- SSL context handler
    if get_ca_certs:
        context.get_ca_certs()
    handler.append(urllib2.HTTPSHandler(context=context))
    #-- redirect handler
    if redirect:
        handler.append(urllib2.HTTPRedirectHandler())
    #-- create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(*handler)
    #-- Encode username/password for request authorization headers
    #-- add Authorization header to opener
    if authorization_header:
        b64 = base64.b64encode('{0}:{1}'.format(username, password).encode())
        opener.addheaders = [("Authorization","Basic {0}".format(b64.decode()))]
    #-- Now all calls to urllib2.urlopen use our opener.
    urllib2.install_opener(opener)
    #-- All calls to urllib2.urlopen will now use handler
    #-- Make sure not to include the protocol in with the URL, or
    #-- HTTPPasswordMgrWithDefaultRealm will be confused.
    return opener

#-- PURPOSE: list a directory on NASA GES DISC https server
def gesdisc_list(HOST, username=None, password=None, build=False,
    timeout=None, urs='urs.earthdata.nasa.gov',
    parser=lxml.etree.HTMLParser(), format='%Y-%m-%d %H:%M',
    pattern='', sort=False):
    """
    List a directory on NASA GES DISC servers

    Parameters
    ----------
    HOST: str or list
        remote https host
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    build: bool, default True
        Build opener with NASA Earthdata credentials
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    urs: str, default 'urs.earthdata.nasa.gov'
        Earthdata login URS 3 host
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for lxml
    format: str, default '%Y-%m-%d %H:%M'
        format for input time string
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    """
    #-- use netrc credentials
    if build and not (username or password):
        username,_,password = netrc.netrc().authenticators(urs)
    #-- build urllib2 opener with credentials
    if build:
        build_opener(username, password, password_manager=True,
            authorization_header=False)
    #-- verify inputs for remote https host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    #-- try listing from https
    try:
        #-- Create and submit request.
        request=urllib2.Request(posixpath.join(*HOST))
        response=urllib2.urlopen(request,timeout=timeout)
    except (urllib2.HTTPError, urllib2.URLError):
        raise Exception('List error from {0}'.format(posixpath.join(*HOST)))
    else:
        #-- read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response,parser)
        colnames = tree.xpath('//tr/td[not(@*)]//a/@href')
        #-- get the Unix timestamp value for a modification time
        lastmod = [get_unix_time(i,format=format)
            for i in tree.xpath('//tr/td[@align="right"][1]/text()')]
        #-- reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(colnames) if re.search(pattern,f)]
            #-- reduce list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            lastmod = [lastmod[indice] for indice in i]
        #-- sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(colnames), key=lambda i: i[1])]
            #-- sort list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            lastmod = [lastmod[indice] for indice in i]
        #-- return the list of column names and last modified times
        return (colnames,lastmod)

#-- PURPOSE: filter the CMR json response for desired data files
def cmr_filter_json(search_results, endpoint="data",
    request_type="application/x-netcdf"):
    """
    Filter the CMR json response for desired data files

    Parameters
    ----------
    search_results: dict
        json response from CMR query
    endpoint: str, default 'data'
        url endpoint type

            - ``'data'``: NASA Earthdata https archive
            - ``'opendap'``: NASA Earthdata OPeNDAP archive
            - ``'s3'``: NASA Earthdata Cumulus AWS S3 bucket
    request_type: str, default 'application/x-netcdf'
        data type for reducing CMR query

    Returns
    -------
    granule_names: list
        Model granule names
    granule_urls: list
        Model granule urls
    granule_mtimes: list
        Model granule modification times
    """
    #-- output list of granule ids, urls and modified times
    granule_names = []
    granule_urls = []
    granule_mtimes = []
    #-- check that there are urls for request
    if ('feed' not in search_results) or ('entry' not in search_results['feed']):
        return (granule_names,granule_urls)
    # descriptor links for each endpoint
    rel = {}
    rel['data'] = "http://esipfed.org/ns/fedsearch/1.1/data#"
    rel['opendap'] = "http://esipfed.org/ns/fedsearch/1.1/service#"
    rel['s3'] = "http://esipfed.org/ns/fedsearch/1.1/s3#"
    #-- iterate over references and get cmr location
    for entry in search_results['feed']['entry']:
        granule_names.append(entry['producer_granule_id'])
        granule_mtimes.append(get_unix_time(entry['updated'],
            format='%Y-%m-%dT%H:%M:%S.%f%z'))
        for link in entry['links']:
            # skip inherited granules
            if ('inherited' in link.keys()):
                continue
            # append if selected endpoint
            if (link['rel'] == rel[endpoint]):
                granule_urls.append(link['href'])
                break
            # alternatively append if selected data type
            if ('type' not in link.keys()):
                continue
            if (link['type'] == request_type):
                granule_urls.append(link['href'])
                break
    #-- return the list of urls, granule ids and modified times
    return (granule_names, granule_urls, granule_mtimes)

#-- PURPOSE: cmr queries for GRACE/GRACE-FO products
def cmr(short_name, version=None, start_date=None, end_date=None,
    provider='GES_DISC', endpoint='data', request_type='application/x-netcdf',
    verbose=False, fid=sys.stdout):
    """
    Query the NASA Common Metadata Repository (CMR) for model data

    Parameters
    ----------
    short_name: str
        Model shortname in the CMR system
    version: str or NoneType, default None
        Model version
    start_date: str or NoneType, default None
        starting date for CMR product query
    end_date: str or NoneType, default None
        ending date for CMR product query
    provider: str, default 'GES_DISC'
        CMR data provider

            - ``'GES_DISC'``: GESDISC
            - ``'GESDISCCLD'``: GESDISC Cumulus
            - ``'PODAAC'``: PO.DAAC Drive
            - ``'POCLOUD'``: PO.DAAC Cumulus
    endpoint: str, default 'data'
        url endpoint type

            - ``'data'``: NASA Earthdata https archive
            - ``'opendap'``: NASA Earthdata OPeNDAP archive
            - ``'s3'``: NASA Earthdata Cumulus AWS S3 bucket
    request_type: str, default 'application/x-netcdf'
        data type for reducing CMR query
    verbose: bool, default False
        print CMR query information
    fid: obj, default sys.stdout
        open file object to print if verbose

    Returns
    -------
    granule_names: list
        Model granule names
    granule_urls: list
        Model granule urls
    granule_mtimes: list
        Model granule modification times
    """
    #-- create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    #-- build urllib2 opener with SSL context
    #-- https://docs.python.org/3/howto/urllib2.html#id5
    handler = []
    #-- Create cookie jar for storing cookies
    cookie_jar = CookieJar()
    handler.append(urllib2.HTTPCookieProcessor(cookie_jar))
    handler.append(urllib2.HTTPSHandler(context=ssl.SSLContext()))
    #-- create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(*handler)
    #-- build CMR query
    cmr_format = 'json'
    cmr_page_size = 2000
    CMR_HOST = ['https://cmr.earthdata.nasa.gov','search',
        'granules.{0}'.format(cmr_format)]
    #-- build list of CMR query parameters
    CMR_KEYS = []
    CMR_KEYS.append('?provider={0}'.format(provider))
    CMR_KEYS.append('&sort_key[]=start_date')
    CMR_KEYS.append('&sort_key[]=producer_granule_id')
    CMR_KEYS.append('&scroll=true')
    CMR_KEYS.append('&page_size={0}'.format(cmr_page_size))
    #-- dictionary of product shortnames and version
    CMR_KEYS.append('&short_name={0}'.format(short_name))
    if version:
        CMR_KEYS.append('&version={0}'.format(version))
    #-- append keys for start and end time
    #-- verify that start and end times are in ISO format
    start_date = isoformat(start_date) if start_date else ''
    end_date = isoformat(end_date) if end_date else ''
    CMR_KEYS.append('&temporal={0},{1}'.format(start_date, end_date))
    #-- full CMR query url
    cmr_query_url = "".join([posixpath.join(*CMR_HOST),*CMR_KEYS])
    logging.info('CMR request={0}'.format(cmr_query_url))
    #-- output list of granule names and urls
    granule_names = []
    granule_urls = []
    granule_mtimes = []
    cmr_scroll_id = None
    while True:
        req = urllib2.Request(cmr_query_url)
        if cmr_scroll_id:
            req.add_header('cmr-scroll-id', cmr_scroll_id)
        response = opener.open(req)
        #-- get scroll id for next iteration
        if not cmr_scroll_id:
            headers = {k.lower():v for k,v in dict(response.info()).items()}
            cmr_scroll_id = headers['cmr-scroll-id']
        #-- read the CMR search as JSON
        search_page = json.loads(response.read().decode('utf8'))
        ids,urls,mtimes = cmr_filter_json(search_page,
            endpoint=endpoint, request_type=request_type)
        if not urls:
            break
        #-- extend lists
        granule_names.extend(ids)
        granule_urls.extend(urls)
        granule_mtimes.extend(mtimes)
    #-- return the list of granule ids, urls and modification times
    return (granule_names, granule_urls, granule_mtimes)

#-- PURPOSE: build requests for the GES DISC subsetting API
def build_request(short_name, dataset_version, url, variables=[],
    format='bmM0Lw', service='L34RS_MERRA2', version='1.02',
    bbox=[-90,-180,90,180], **kwargs):
    """
    Build requests for the GES DISC subsetting API

    Parameters
    ----------
    short_name: str
        Model shortname in the CMR system
    url: str
        url for granule returned by the CMR system
    variables: list, default []
        Variables for product to subset
    format: str, default 'bmM0Lw'
        Coded output format for GES DISC subsetting API
    service: str, default 'L34RS_MERRA2'
        GES DISC subsetting API service
    version: str, default '1.02'
        GES DISC subsetting API service version
    bbox: list, default [-90,-180,90,180]
        Bounding box to spatially subset
    **kwargs: dict, default {}
        Additional parameters for GES DISC subsetting API

    Returns
    -------
    request_url: str
        Formatted url for GES DISC subsetting API
    """
    #-- split CMR supplied url for granule
    HOST,*args = url_split(url)
    api_host = posixpath.join(HOST,'daac-bin','OTF','HTTP_services.cgi?')
    #-- create parameters to be encoded
    kwargs['FILENAME'] = posixpath.join(posixpath.sep, *args)
    kwargs['FORMAT'] = format
    kwargs['SERVICE'] = service
    kwargs['VERSION'] = version
    kwargs['BBOX'] = ','.join(map(str, bbox))
    kwargs['SHORTNAME'] = short_name
    kwargs['DATASET_VERSION'] = dataset_version
    kwargs['VARIABLES'] = ','.join(variables)
    #-- return the formatted request url
    request_url = api_host + urlencode(kwargs)
    return request_url
