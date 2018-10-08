import getopt
import gzip
import os
import re
import sys
import tarfile
from UserDict import UserDict
from cStringIO import StringIO
from glob import glob
from subprocess import Popen

from paths import paths


def ar2tar(ar,fname,tarh):
	arstr = ar.tostring()
	info = tarfile.TarInfo(fname)
	info.size = len(arstr)
	tarh.addfile(info,StringIO(arstr))

def tar2ar(fname,tarh,SHAPE,dtype=float):
	arstr = tarh.extractfile(tarh.getmember(fname)).read()
	ar = numpy.fromstring(arstr,dtype=dtype).reshape(SHAPE)
	return ar

def obj2tar(obj,fname,tarh):
	objstr = obj.__repr__()
	info = tarfile.TarInfo(fname)
	info.size = len(objstr)
	tarh.addfile(info,StringIO(objstr))

def tar2obj(fname,tarh,eval_fn=eval):
	objstr = tarh.extractfile(tarh.getmember(fname)).read()
	obj = eval_fn(objstr)
	return obj

def append_ar2tar(ar,fname,tarf):
	tarh = tarfile.open(tarf,'a')
	ar2tar(ar,fname,tarh)
	tarh.close()

def append_obj2tar(obj,fname,tarf):
	tarh = tarfile.open(tarf,'a')
	obj2tar(obj,fname,tarh)
	tarh.close()

def read_tar2ar(fname,tarf,SHAPE,dtype=float):
	tarh = tarfile.open(tarf,'r')
	ar = tar2ar(fname,tarh,SHAPE,dtype)
	tarh.close()
	return ar

def read_tar2obj(fname,tarf,eval_fn=eval):
	tarh = tarfile.open(tarf,'r')
	obj = tar2obj(fname,tarh,eval_fn)
	tarh.close()
	return obj

def tar2ar_all(tarh,SHAPE,dtype=float):
	return [tar2ar(f,tarh,SHAPE,dtype) for f in sorted(tarh.getnames())]

def tar2obj_all(tarh,eval_fn=eval):
	return [tar2obj(f,tarh,eval_fn) for f in sorted(tarh.getnames())]


#####
# command line parsing
#####

def shortopts_from_longopts(longopts):
	'''
	**getopt_long below should render this unnecessary!**
	
	given a string of long options, returns 2-tuple:
	
	( arg string for single-letter shortcuts,
	  dictionary of short : long
	'''
	
	lodict = {}
	shortopts = ''
	
	for opt in longopts:
		pick = None
		for candidate in opt.rstrip('='):
			if candidate not in lodict.keys():
				pick = candidate
				break
		if pick:
			lodict[pick] = opt.rstrip('=')
			so = pick
			if opt.endswith('='):
				so += ':'
			shortopts += so
		else:
			raise ValueError, 'No valid short option for %s (current picks: %s)' % (opt,lodict)
			
	return shortopts,lodict
		
def getopt_long(args,optstruct,required=None,help='h'):
	'''given a special dict containing:
	
	long_option_name : ( single_letter, typing_function, default, helpstring )
	
	handles construction of a dict:
	
	long_option_name : typing_function(value_from_argv) 
	
	and returns usage statement upon parser error
	
	for true/false flags, use 'flag' / 'unflag' as typing_function, set default state w/ default  
	
	ex. from analyze_antfarm.py:
	#<snip>
	optstruct = { 
		'seglen': ('s',int,30,'length of analysis segment in seconds'),
		'mask' : ('m',str,'/n/home/brantp/code/video_analysis/nomask','filename of a dimension-matched binary mask file'),
		'burrow_mask' : ('k',str,None,'filename of dimension-matched binary burrow mask file [deprecated]'),
		'burrow_entrance_xy' : ('b',str,'"(360,240)"','x,y tuple of burrow entrance coordinates'),
		'ground_start' : ('g',int,None,'approx y value of ground level at left edge of xybounds'),
		'cleanup' : ('c','flag',False,'delete images when finished.  removes whole image tree (i.e. rm -rf imagedir) so use with caution!'),
		#etc etc
		}
		
	opts,args = Util.getopt_long(sys.argv[1:],optstruct,required=['ground_start'])
	#</snip>
	'''
	
	def usage(args,optstruct):
		from pprint import pprint
		print >> sys.stderr,'USAGE:\n'
		pprint(optstruct,sys.stderr)
		print >> sys.stderr,'\n\nRECEIVED:\n'
		pprint(args,sys.stderr)
		
	def flag(val):
		return True

	def unflag(val):
		return False

	shortopts = ''
	longopts = []
	solookup = {}
    #if help is any single character, add a help option to catch later
	if isinstance(help,str) and len(help) == 1:
		longopts.append('help')
		shortopts += help
		solookup[help] = 'help'

	
	for lopt,(sopt,tf,default,helpstr) in optstruct.items():
		if 'flag' in [tf, getattr(tf,'__name__',None)]:
			optstruct[lopt] = (sopt,flag,default,helpstr)
			shortopts += sopt
			longopts.append(lopt)
		elif 'unflag' in [tf, getattr(tf,'__name__',None)]:
			optstruct[lopt] = (sopt,unflag,default,helpstr)
			shortopts += sopt
			longopts.append(lopt)
		else:
			shortopts += sopt+':'
			longopts.append(lopt+'=')
		solookup[sopt] = lopt
	
	try:
		opts,arguments = getopt.getopt(args,shortopts,longopts)
	except getopt.GetoptError:
		usage(args,optstruct)
		raise
	
	parsedopts = {}
	for opt,val in opts:
		if opt.startswith('--'):
			key = opt[2:]
		else:
			key = solookup[opt[1:]]
			
		#check for specific 'help' option invocation if set
		if isinstance(help,str) and len(help) == 1:
			if key == 'help':
				usage(args,optstruct)
				sys.exit(2)
		
		if parsedopts.has_key(key):
			usage(args,optstruct)
			raise ValueError, 'option %s present more than once!'
		else:
			parsedopts[key] = optstruct[key][1](val)
			
	#set defaults
	for lopt,(sopt,tf,default,helpstr) in optstruct.items():
		if not parsedopts.has_key(lopt):
			if default is not None:
				parsedopts[lopt] = default
			
	if required:
		if not all([req in parsedopts.keys() for req in required]):
			usage(args,optstruct)
			raise ValueError, 'not all required options present (required: %s)' % required
			
	return parsedopts,arguments
			
#####
# misc util
#####
import collections


def collate_files_by_path(files,outroot,verbose=True):
	'''given a list of files, generates files in outroot
	with names composed of all non-shared path elements

	e.g.
	thing/stuff/20times/boring/ving.pdf
	thing/stuff/10times/boring/ving.txt
	thing/nonstuff/20times/boring/ving.pdf

	will result in the files
	stuff_20times_ving.pdf
	stuff_10times_ving.txt
	nonstuff_20times_ving.pdf

	
	'''
	try:
		os.makedirs(outroot)
	except:
		pass

	pathels = collections.defaultdict(list)
	for f in files:
		[pathels[i].append(el) for i,el in enumerate(f.split('/'))]
	#print pathels

	keeps = [i for i,vals in sorted(pathels.items())[:-1] if len(set(vals)) != 1]
	keeps.append(sorted(pathels.items())[-1][0]) #always keep the filename itself
	
	for f in files:
		newf = os.path.join(outroot,'__'.join(numpy.array(f.split('/'))[keeps]))
		if verbose: print >> sys.stderr, '%s -> %s' % (f,newf)
		ret = os.system('cp %s %s' % (f,newf))
		if verbose: print >> sys.stderr, ret

def file_in_path(filename,pathname):
	'''given a path, finds the instance of filename farthest "leaf"-ward'''

	filetry = os.path.join(pathname,filename)

	if os.path.exists(filetry):
		return filetry
	else:
		if pathname == '/':
			raise ValueError, '%s not present in original search path' % (filename)
		elif not '/' in pathname:
			return file_in_path(filename,os.getcwd())
		else:
			return file_in_path(filename,os.path.dirname(pathname))
		
import random, string
def random_filename(chars=string.hexdigits, length=8, prefix='', suffix='', \
                        verify=True, attempts=10):
    for attempt in range(attempts):
        filename = ''.join([random.choice(chars) for i in range(length)])
        filename = prefix + filename + suffix
        if not verify or not os.path.exists(filename):
            return filename

ERROR_STR= """Error removing %(path)s, %(error)s """

def sizeof(obj):
    """APPROXIMATE memory taken by some Python objects in 
    the current 32-bit CPython implementation.

    Excludes the space used by items in containers; does not
    take into account overhead of memory allocation from the
    operating system, or over-allocation by lists and dicts.
    """
    T = type(obj)
    if T is int:
        kind = "fixed"
        container = False
        size = 4
    elif T is list or T is tuple:
        kind = "variable"
        container = True
        size = 4*len(obj)
    elif T is dict:
        kind = "variable"
        container = True
        size = 144
        if len(obj) > 8:
            size += 12*(len(obj)-8)
    elif T is str:
        kind = "variable"
        container = False
        size = len(obj) + 1
    else:
        raise TypeError("don't know about this kind of object")
    if kind == "fixed":
        overhead = 8
    else: # "variable"
        overhead = 12
    if container:
        garbage_collector = 8
    else:
        garbage_collector = 0
    malloc = 8 # in most cases
    size = size + overhead + garbage_collector + malloc
    # Round to nearest multiple of 8 bytes
    x = size % 8
    if x != 0:
        size += 8-x
        size = (size + 8)
    return size


def cat(filelist,targetfile):
    '''cats an arbitrarily large filelist to targetfile'''
    fh = open(targetfile,'w')
    for f in filelist:
        fh.write(open(f).read()+'\n')
    fh.close()

def hamming(s1,s2):
	hamm = 0
	for i in xrange(len(s1)):
		if s1[i] != s2[i]:
			hamm += 1
	return hamm

import itertools
def mode(li):
    '''return modal value for a list of numbers'''
    return sorted([(len(list(g)),k) for k,g in itertools.groupby(sorted(li))],reverse=True)[0][1]

def zscore(values_in):
    '''returns a list of z-score transformed values'''
    import numpy
    if type(values_in) == numpy.array:
        values = values_in
    else:
        values = numpy.array(values_in)

    if (values > 0).any():
        zvals = ( values - values.mean() ) / values.std()
    else:
        return values_in
        

    if isinstance(values_in, list):
        return list(zvals)
    else:
        return zvals

def zscore_local(values_in,win=100):
	import numpy
	values = numpy.array(values_in)

	halfwin = int(win/2)

	if (values != 0).any():
		zvals = numpy.array([zscore(values[i-halfwin:i+halfwin])[halfwin] for i in range(halfwin,len(values)-halfwin)])
	else:
		return values_in
	
	
	if isinstance(values_in, list):
		return list(zvals)
	else:
		return zvals

	

def interpolate_missing_values(values_in,missing=None,progress=False):
	'''performs linear interpolation of list elements in partially numerical list
	treats items matching <missing> as values to infer
	
	ex:
	>>>Util.interpolate_missing_values([0,1,2,None,4])
	array([1, 2, 3, 4], dtype=object)
	
	'''
	import numpy
	values = numpy.array(values_in)
	tot = float(len(values))
	lastpct = 0
	values[0] = values[values>0][0]
	values[-1] = values[::-1][values[::-1]>0][0]
	for i,k in enumerate(values):
		pct = int(i / tot*100.0)
		if progress and pct != lastpct and pct % 1 == 0:
			print >> sys.stderr, pct
			lastpct = pct
		if k == missing:
			revar = values[i:0:-1]
			fwdar = values[i:]
                        try:
                            d_rev = list(revar > 0).index(True)
                            v_rev = revar[revar > 0][0]
                        except ValueError:
                            d_rev = i
                            v_rev = 0
                        try:
                            d_fwd = list(fwdar > 0).index(True)
                            v_fwd = fwdar[fwdar > 0][0]
                        except ValueError:
                            d_fwd = len(values) - i
                            v_fwd = v_rev

                        if v_rev == 0:
                            v_rev = v_fwd
			values[i] = v_rev + int((v_fwd-v_rev) * (d_rev/float(d_rev+d_fwd)))
	return values

def subtract_mask(tmask,submask,newvalue=False):
	'''flips true values in tmask to false for all 'true' values in submask 
	hax: can set elements corresponding to anything that evaluates to "true" in submask to arbitrary (<newvalue>) in tmask'''	

	if tmask.shape != submask.shape:
		raise ValueError, 'shapes incompatible'
	shape = tmask.shape
	tflat = tmask.flatten()
	for i,k in enumerate(submask.flatten()):
		if k:
			tflat[i] = False
	return tflat.reshape(shape)

def smoothListGaussian(list,degree=5):
	window=degree*2-1
	weight=numpy.array([1.0]*window)
	weightGauss=[]
	for i in range(window):
		i=i-degree+1
		frac=i/float(window)
		gauss=1/(numpy.exp((4*(frac))**2))
		weightGauss.append(gauss)
	weight=numpy.array(weightGauss)*weight
	smoothed=[0.0]*(len(list)-window)
	for i in range(len(smoothed)):
	       	smoothed[i]=sum(numpy.array(list[i:i+window])*weight)/sum(weight)
	return smoothed

def gauss_kern(size, sizey=None):
	""" Returns a normalized 2D gauss kernel array for convolutions """
	size = int(size)
	if not sizey:
		sizey = size
	else:
		sizey = int(sizey)
	x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
	g = numpy.exp(-(x**2/float(size)+y**2/float(sizey)))
	return g / g.sum()

def gauss_smooth(im, n, ny=None):
	""" blurs the image by convolving with a gaussian kernel of typical
	size n. The optional keyword argument ny allows for a different
	size in the y direction.
	"""
	from scipy import signal
	g = gauss_kern(n, sizey=ny)
	improc = signal.convolve(im,g, mode='valid')
	return(improc)


def smooth(values_in,window=50,pad_flat=True,interpolate_nones=False,exact=False,pad_value=None):
	'''smooths an array over window size window
	
	will, by default, 'pad' the array back to its original size.
	will use pad_value if set, otherwise:
	if the array is 1-D, pads will be the value at each end, 
	if 2-D, pads with lowest value in the array
	
	2-D only: if exact is true, will perform block average smoothing (very slow)
	otherwise averages in 'stripes' which is much faster, but not provably "right"'''
	
	import numpy
	values = numpy.array(values_in)

	window = window - (window % 2)
	halfwin = window/2
	if len(values.shape) == 1:
		if interpolate_nones:
			values = interpolate_missing_values(values)
		l = [values[i-halfwin:i+halfwin].mean() for i in range(halfwin,len(values)-halfwin)]
		if pad_flat:
			if pad_value:
				l = [pad_value]*halfwin + l + [pad_value]*halfwin
			l = l[:1]*halfwin + l + l[-1:]*halfwin
		return numpy.array(l)
	elif len(values.shape) == 2:
		if pad_flat:
			if pad_value is None:
				pad_value = values.min()
			ar = numpy.zeros(values.shape)
			ar[:,:] = pad_value
			ar[halfwin:halfwin*-1,halfwin:halfwin*-1] = smooth(values,window,pad_flat=False,exact=exact)
			return ar
		if exact:
			return numpy.array([[values[i-halfwin:i+halfwin,j-halfwin:j+halfwin].mean() for j in xrange(halfwin,len(values[0])-halfwin)] for i in xrange(halfwin,len(values)-halfwin)])
		else:
			convol = [values_in[window-i:-i,halfwin:-halfwin] for i in xrange(1,window)] \
					+[values_in[halfwin:-halfwin,window-i:-i] for i in xrange(1,window)] \
					+[values_in[window:,halfwin:-halfwin],values_in[halfwin:-halfwin,window:]]
			denom = numpy.zeros(convol[0].shape)
			denom[:,:] = len(convol)
			return reduce(lambda x,y: x+y, convol) / denom
	else:
		raise ValueError, 'Currently only 1-D and 2-D smoothing implemented'

def zsmooth_loop(mat3d,win):
	'''slow 3D smoothing'''
	tsm = numpy.zeros(mat3d.shape)
	for i in range(mat3d.shape[-2]):
		for j in range(mat3d.shape[-1]):
			tsm[:,i,j] = smooth(mat3d[:,i,j],win)
	return tsm
	
def zsmooth_convol(mat3d,win,report_progress=sys.stderr):
	'''fast, memory-hungry 3D smoothing'''
	tsm = numpy.zeros(mat3d.shape,dtype=float)
	halfwin = int(win/2)
	step = len(mat3d)/10
	for i in xrange(halfwin,len(mat3d)-halfwin):
		for m in mat3d[i-halfwin:i+halfwin]:
			tsm[i] += m
		if report_progress and i % step == 0:
			print >> report_progress, 'frame %s done' % i
	tsm /= win
	for i in range(halfwin):
		tsm[i] = tsm[halfwin]
		tsm[-1-i] = tsm[-1-(halfwin)]
	return tsm
	
def normalize(values_in,min=0,max=1.0,axis=None,to_abs=False,set_max=None):
    '''normalizes an array() - able set of values 'values_in'
    if max/min are used, spreads the values between them

    returns the type it got (array, list, tuple)

    axis=1 means normalize each row; set to axis=0 
    if column normalization in desired
    '''
    import numpy
    if set_max:
        values = numpy.array([v>set_max and set_max or v for v in values_in]+[set_max])
    else:
        values = numpy.array(values_in)
	    
    if to_abs:
        values = numpy.absolute(values)

    if len(values.shape) > 1 and not axis is None:
        curr_min = values.min(axis=axis).reshape((-1, 1))
        curr_max = values.max(axis=axis).reshape((-1, 1))
    else:
        curr_min = values.min()
        curr_max = values.max()
    
    if curr_max > curr_min:
        values = values - curr_min

    if len(values.shape) > 1 and not axis is None:
        curr_max = values.max(axis=axis).reshape((-1, 1))
    else:
        curr_max = values.max()
    values = values / float(curr_max)

    values = values * (float(max) - min)
    values = values + min

    if set_max:
        values = numpy.array(list(values)[:-1])
    if isinstance(values_in,list):
        values = [i for i in values]
    elif isinstance(values_in,tuple):
        values = tuple([i for i in values])

    return values

def get_consecutive_value_boundaries(li):
    '''returns (start,end) tuples for all consecutive elements 
    >>> get_consecutive_value_boundaries([ 0,  1,  2,  6,  7,  8, 12, 15, 16, 17])
    [(0, 2), (6, 8), (12, 12), (15, 17)]'''
    if len(li) < 2:
        return []
    ar = numpy.array(li)
    ar.sort()
	
    starts = [ar[0]]+list((ar[1:])[(ar[1:] - ar[:-1]) > 1])
    ends = list(ar[(ar[1:] - ar[:-1]) > 1] + 1)+[ar[-1] + 1]
    return zip(starts,ends)

def coord_tuple_from_flat_index(idx,shape):
	'''given an index in a flattened array and the shape of the original, returns corresponding coord tuple
	
	JUST USE xy_whateverwhatever FROM VIDTOOLS'''
	pass
	
def sec_from_hms(hms):
	'''given an hours:min:sec string, returns integral seconds'''

	if hms.startswith('-'):
		coef = -1
		hms = hms[1:]
	else:
		coef = 1
	
	h,m,s = [int(i) for i in hms.split(':')]
	return coef * ((h*3600)+(m*60)+s)

def hms_from_sec(sec):
	return '%s:%02d:%02d' % (sec/3600,sec/60%60,sec%60)

def line_fx_from_pts(p1,p2):
	'''returns a FUNCTION for a line through two points supplied'''
	run,rise = numpy.array(p1)-numpy.array(p2)
	m = rise/run
	b = p1[1]-(m*p1[0])
	
	return lambda x: m*x+b

def multiple_domain_fx_from_pts(pts):
	'''given a list of x,y points,

	returns a FUNCTION that supplies a y value on the lines linking those points for any given x'''

	if len(pts) == 2:
		return line_fx_from_pts(*pts)

	pts.sort()

	fx = [line_fx_from_pts(p,pts[i+1]) for i,p in enumerate(pts[:-1])]
	breaks = [x for x,y in pts[1:-1]]
	def this_fx(x):
		these_breaks = breaks
		these_fx = fx
		#logic here to choose fit regieme and return approrpiate y
		if x < these_breaks[0]:
			return(these_fx[0](x))
		elif x > these_breaks[-1]:
			return(these_fx[-1](x))
		else:
			for brk,f in zip(these_breaks[1:],these_fx[1:-1]):
				if x < brk:
					return f(x)

	return this_fx
	
		

def dezip(values_in):
	'''opposite of zip(), i.e. 
	>>> dezip([('a',1),('b',2),('c',3)])
	(['a','b','c'],[1,2,3])'''

        if isinstance(values_in,tuple):
            values_in = [values_in]
	
	#return [tuple([i[0] for i in values_in if len(i) == 2]),tuple([i[1] for i in values_in if len(i) == 2])]
	lol = []
	for i in range(len(values_in[0])):
		lol.append([])
	for l in values_in:
		for it,li in zip(l,lol):
			li.append(it)
	return tuple(lol)

def split_on_value(values_in,splitfn = lambda x: x is None):
	'''given a list, splits into lists breaking on values which satisfy splitfn
	by default, splits on None values
	'''

	lol = []
	these = []
	for val in values_in:
		if splitfn(val):
			if these: lol.append(these)
			these = []
		else:
			these.append(val)
	if these: lol.append(these)
	return lol
			

def merge_dictlist(dicts,verbose=False):
	'''combines a list of dictionaries into a single dictionary
	if a given key appears in multiple dictionaries, no telling what might happen! '''
	#return reduce(lambda x,y: dict(x.items()+y.items()), dicts)
	mergeli = []
	if verbose:
		print >> sys.stderr, 'merging %i dicts' % len(dicts)
	for di in dicts:
		mergeli.extend(di.items())
	if verbose:
		print >> sys.stderr, 'merge complete, covert to dict'
	return dict(mergeli)

def countdict(li):
    '''return a dictionary where keys are unique list values and values are counts of items of that value'''
    d = {}.fromkeys(set(li),0)
    for l in li:
        d[l] += 1
    return d

def invert_dict(di):
    '''returns a dictionary where keys are values from di, values are lists of keys from di w/ that value

    e.g.
    >>> di = {"thing":0,"other":1,"also":0}
    >>> invert_dict(di)
    { 0:["thing","also"], 1:["other"] }

    Note that all kinds of things will break this!
    (like mutable values in di, for instance)
    '''
    d = {}.fromkeys(set(di.values()),None)
    for k in d.keys():
        d[k] = []
    for k,v in di.items():
        d[v].append(k)
    return d

def flatten_list(li):
	return reduce(lambda x,y:x+y, li)
	
def subdict_from_list(di,li):
	'''given a dictionary and a list of keys, returns a new dict composed only of items for those keys'''
	return dict([(k,v) for k,v in di.items() if k in li])

def parse_tabular(filename,header_key='#',delim='\t'):
    '''parse a tabular file with a header row starting w/ header_key

    return list of dicts (one dict per line, key:value per field in that record)'''

    re_str = None
    lod = []
    
    for l in open(filename):
        if l.startswith(header_key):
            if re_str is None:
                fields = l.strip().lstrip(header_key).split(delim)
                re_str = delim.join([r'(?P<%s>.+?)' % f for f in fields])
                re_str = '^'+re_str+'$'
	else:
            match = re.search(re_str,l)
            if match:
                lod.append(match.groupdict())
    return lod

def rmgeneric(path, __func__):

    try:
        __func__(path)
        #print 'Removed ', path
    except OSError, (errno, strerror):
        print ERROR_STR % {'path' : path, 'error': strerror }
            
def removeall(path):

    if not os.path.isdir(path):
        return
    
    files=os.listdir(path)

    for x in files:
        fullpath=os.path.join(path, x)
        if os.path.isfile(fullpath):
            f=os.remove
            rmgeneric(fullpath, f)
        elif os.path.isdir(fullpath):
            removeall(fullpath)
            f=os.rmdir
            rmgeneric(fullpath, f)

def load_list_from_dir_contents(di,ty=None):
    '''loads contents of all files in di.
    if ty specified, attempts to type each element on load.'''

    li = []
    for f in os.listdir(di):
        li.extend(open(os.path.join(di,f)).readlines())
    if ty:
        li = [ty(i) for i in li]
    return li

def eval_from_dir(tdir,globst='*',ty=eval,verbose=False):
	flist = sorted(glob(os.path.join(tdir,globst)))
	rlist = []
	for f in flist:
		if verbose: print >> sys.stderr, f
		cont = open(f).read()
		rlist.append(ty(cont))
	return rlist

def smartopen(filename,*args,**kwargs):
    '''opens with open unless file ends in .gz, then use gzip.open

    in theory should transparently allow reading of files regardless of compression
    '''
    if filename.endswith('.gz'):
        return gzip.open(filename,*args,**kwargs)
    else:
        return open(filename,*args,**kwargs)


import numpy
class AssociativeArray(dict):
    '''creates a 2D numpy array with cols (inner) = length, and rows (outer) = len(keys_list)

    retrieval of a key gives the row of the array

    example:

    >>> aa = Util.AssociativeArray(['dmel','dere','dpse'],20,int)
    >>> aa['dere']

    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    '''

    def __init__(self,keys_list,length,creator=numpy.zeros,**creator_args):
        '''creator is any function that will take a dimensionality tuple and arbitrary kwargs to return a numpy.array'''
        dict.__init__(self)
        self.array = creator((len(keys_list),length),**creator_args)
        self.keys_list = keys_list
        for i,k in enumerate(keys_list):
            self[k] = i

    def __getitem__(self,key):
        idx = dict.__getitem__(self,key)
        return self.array[idx]
    
    def idx(self,key):
        return dict.__getitem__(self,key)

#HACK LSF IMPORTS
try:
	from mice_behavior_capstone.drafting import lsf_jobs_dict
	from mice_behavior_capstone.drafting import lsf_get_job_details
	from mice_behavior_capstone.drafting import lsf_jobs_submit
	from mice_behavior_capstone.drafting import lsf_jobs_status
	from mice_behavior_capstone.drafting import lsf_wait_for_jobs
	from mice_behavior_capstone.drafting import lsf_kill_jobs
	from mice_behavior_capstone.drafting import lsf_restart_jobs
	from mice_behavior_capstone.drafting import lsf_restart_susp
except ImportError:
	pass

#####
# Controller stuff; mostly deprecated/defunct
#####       

class Controller (UserDict):
    '''generic object for subprocess control

    should be subclassed to manage specific apps'''

    controller_type = ''

    def __init__(self,tempdir=None,stdin=None,stderr=None,stdout=None,
                 executor=Popen,use_defIOE=0,name=None,cleanup=0,args=None,logfile=None):
        '''main functionality is to determine default temp dir and I/O/E settings

        '''
        UserDict.__init__(self)

            
        self["executor"] = executor
        self['cleanup'] = cleanup
        self['files'] = []
        self['logfile'] = logfile
        self['out'] = None

        if args:
            self.controller_type = os.path.basename(args[0])
            self['args'] = args

        if name:
            self["name"] = name
        else:
            self["name"] = random_filename(prefix=self.controller_type)

        if tempdir: #no longer calls setitem, must do this manually, or in compose_arg
            self.data["tempdir"] = tempdir
        else: #not explicitly set by call; wait on dir creation
            prog = sys.argv[0] and sys.argv[0] or self.__module__
            self.data["tempdir"] = os.path.join(paths["temp"],os.path.basename(prog))

        #if use_defIOE=1 set stdin, stderr and stdout based on self.default_IOE(),
        #otherwise use individual arguments
        #since assignment is to self.data instead of self, __setitem__ behaviors do not fire
        (self.data["stdin"],self.data["stdout"],self.data["stderr"]) = \
            use_defIOE and self.default_IOE() or (stdin,stdout,stderr)


    def default_IOE(self):
        '''returns 3-tuple (stdin, stdout, stderr) based on current values of   
        self['name'] and self['tempdir']

        only called if use_defIOE=1'''
        default_IOE = (None, ".stdout", ".stderr")
        return (None, 
                os.path.join(self["tempdir"],self["name"]+'.stdout'),
                os.path.join(self["tempdir"],self["name"]+'.stderr') )

    def setupIO(self):
        '''
        should take over the job of opening fh instances of stdin, stderr stdout as necessary

        '''
        pass

    def __getitem__(self, key):
        if key == "tempdir":
            os.path.exists(self.data["tempdir"]) \
                or os.makedirs(self.data["tempdir"])
        return UserDict.__getitem__(self, key)

    def __setitem__(self, key, item):
        UserDict.__setitem__(self, key, item)
        if key == "tempdir" and item:
            if not os.path.exists(self.data["tempdir"]):
                os.makedirs(self.data["tempdir"])
        if key in ["stdout","stderr"] and isinstance(item, str):
            self.data[key] = open(item,'a')
            self['files'].append(item)
        if key == 'stdin' and isinstance(item,str):
            self.data[key] = open(item)

    def run(self,now=0,cwd=None):
        print >> sys.stderr, "calling %s for stdo %s stde %s\n" % (self["executor"],self["stdout"],self["stderr"])
        if self['logfile']: open(self['logfile'],'a').write('starting run\n'
                                                            "calling %s for stdo %s stde %s\n" % \
                                                                (self["executor"],self["stdout"],self["stderr"]))
	self['tempdir'] = self.data['tempdir']
        if cwd is None: 
            cwd = self["tempdir"]


        #assigning via __setitem__ spins up relevant fh instances
        (self["stdin"],self["stdout"],self["stderr"]) = \
            (self.data["stdin"],self.data["stdout"],self.data["stderr"])

        args = self.compose_arguments()
        print >> sys.stderr, "running %s in %s" % (' '.join(args),cwd)
        if self['logfile']: open(self['logfile'],'a').write('finished run\n'
                                                            "%s in %s" % (' '.join(args),cwd))
        if now:
            #added self['out'] and changed .wait() to .communicate(); hope this doesn't break!
            self['out'] = self["executor"](args,
                                           stdin=self['stdin'],
                                           stdout=self['stdout'],
                                           stderr=self['stderr'],
                                           cwd=cwd).communicate()
            return self.handle_results()
        else:
            return self["executor"](args,
                                    stdin=self['stdin'],
                                    stdout=self['stdout'],
                                    stderr=self['stderr'],
                                    cwd=cwd)

    def compose_arguments(self):
        '''should be replaced in descendents
        '''
        return self['args']

    def handle_results(self):
        '''at minimum, removes files in self['files'] if cleanup=1
        '''
        if self['cleanup']: self.cleanup()

    def cleanup(self):
        for f in ['stdin','stdout','stderr']:
            if isinstance(self[f],file):
                #print "closing %s" % self[f]
                self[f].close()
        for f in self['files']:
            try :
                os.unlink(f)
            except OSError:
                pass        
        
    
        #removeall(self.data['tempdir'])
        #print os.listdir(self.data['tempdir'])
        #os.rmdir(self.data['tempdir'])

class QueueError(Exception): pass

class Dispatcher():
    '''runs Controller instances on a queue

    queue items (stored in self.queue) are a 3 item dict: {controller, runner, prereqs}

    prep_controller may be set with a function, and if used will be run on each item, passing the item and self
    before the item is run

    TODO: separate total run number from finished in this round (i.e. in this run_controllers() call)

    '''

    def __init__(self,controllers=[],max_queue=100,max_running=100,sleep_time=20,keep_finished=False,prep_controller=None,verbose=True):
        self.queue = []
        self.running = []
        self.done = set()
        self.keep_finished = keep_finished
        self.finished_items = {}
        self.returns = {}
        self.max_queue=max_queue
        self.max_running=max_running
        self.sleep_time = sleep_time
        self.prep_controller = prep_controller
        self.verbose = verbose
        for c in controllers:
            self.add_controller(c)
        

    def add_controller(self, controller, prereqs=None):
        '''takes either a single controller and an optional set of prereq job names (controller['name'])
        or a tuple of controllers in order of required execution.

        if the latter, queues each with the previous as prerequisite'''

        #next part assumes prereqs are set; this is just lazy-user cleanup
        if prereqs is not None and not isinstance(prereqs,set):
            if isinstance(prereqs,list) or isinstance(prereqs,tuple):
                prereqs = set(prereqs)
            else:
                prereqs = set([prereqs])

        if isinstance(controller,tuple):
            for i,c in enumerate(controller):
                if i > 0:
                    self.add_controller(c,controller[i-1]['name'])
                elif i == 0:
                    self.add_controller(c,prereqs)
        else:
            if self.verbose: print >> sys.stderr, 'adding',controller['name']
            job_d = {'controller':controller,'runner':None,'prereqs':prereqs}
            self.queue.append(job_d)  

        if len(self.queue) > self.max_queue: self.run_controllers()

    def next_valid_item(self):
        '''returns the next (i.e. lowest index) controller on queue with all prereqs finished
        
        NB: pops item from list, so the return _must_ be handled, or the controller is lost

        '''

        if len(self.queue) == 0: return None

        for i,item in enumerate(self.queue):
            if item['prereqs'] is None or item['prereqs'].issubset(self.done):
                if self.verbose: print >> sys.stderr, item['controller']['name'],'is next up'
                return self.queue.pop(i)

        #if for loop completes without hitting a valid item and self.running is empty, raise error
        if len(self.running) == 0:
            raise QueueError, 'no valid queue items, nothing running; queue len:',len(self.queue)

    def handle_completed_item(self,item):

        if self.verbose: print >> sys.stderr, 'dealing with completed job:',item['controller']['name']
        self.returns[item['controller']['name']] = item['controller'].handle_results()
        self.done.add(item['controller']['name'])
        if self.keep_finished: self.finished_items[item['controller']['name']] = (item)

    def clear_completed_from_running(self):

        for i,item in enumerate(self.running):
            if item['runner'].poll() == 0:
                if self.verbose: print >> sys.stderr, item['controller']['name'],'done'
                self.handle_completed_item(self.running.pop(i))

    def run_controllers(self):
        '''controllers in this case are a 3 item dict: {controller, runner, prereqs}

        run_controllers will start (i.e. call the controller's run() method and store the resulting subprocess runner)
        valid queued controllers until max_running is reached, and keep running controllers until queue is empty
        '''
        total = len(self.queue)
        import time
        while self.queue or self.running:
            self.clear_completed_from_running()

            while (len(self.running) < self.max_running) and (len(self.queue) > 0):
                next = self.next_valid_item()
                if next is None:
                    if self.verbose: print >> sys.stderr, 'no jobs currently valid'
                    break
                else:
                    if self.prep_controller:
                        self.prep_controller(next,self)
                    next['runner'] = next['controller'].run()
                    self.running.append(next)

            print >> sys.stderr, '%d done, %d running, %d remain of %d total' \
                    % (len(self.done),len(self.running),len(self.queue),total)

            if self.running: time.sleep(self.sleep_time)





if __name__ == "__main__":
    test = Controller(use_defIOE=1,cleanup=0)
    print test
    tdir = test['tempdir']
    print tdir
    print os.path.exists(tdir)
    test.handle_results()
    del test
    print os.path.exists(tdir)
