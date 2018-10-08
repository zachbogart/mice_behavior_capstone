try:
    from shapely.geometry import MultiPoint,Polygon
except:
    import warnings
    warnings.warn('shapely import failed (likely pylab conflict); shapely features unavailable')
    
#try to use a non-interactive backend, as this speeds plot generation
import matplotlib
matplotlib.use('Agg') #will fail with warning if another backend is already loaded; this is fine
import pylab

from PIL import Image,ImageFilter,ImageDraw
from glob import glob
import os, sys, re, Util, LSF, numpy, subprocess, shutil, time, io
from collections import defaultdict
try:
    import cv
    cv_found = True
except:
    print >> sys.stderr, "opencv bindings import failed; disabled"
    cv_found = False

###################################
# OPENCV HELPERS
###################################

SMOOTH5 = ImageFilter.Kernel( (5, 5) , (1,2,2,2,1,\
					2,3,3,3,2,\
					2,3,5,3,2,\
					2,3,3,3,2,\
					1,2,2,2,1 ) )

def array_from_stream(stream,img_smooth_kernel=None,normed=True):
    img = cv.QueryFrame(stream)
    pi = Image.fromstring("RGB", cv.GetSize(img), img.tostring()).convert(mode="L")
    if img_smooth_kernel:
        pi = pi.filter(img_smooth_kernel)
    m = numpy.asarray(pi)
    if normed:
       	m = Util.normalize(m)
    return m

def mat_polys2cv(m,polys=[],hatchpolys=[],lines=[],dpi=80,scale=1,bars=[]):
    '''NOTE: differs from initial version in that polys is now (color,polygon_pts) tuples list
    for a bare list of polygon points <pp>, consider calling with polys=iplot.subspec_enum(pp)
    hatchpolys is 3-tuple (color,hatch,poly)
    '''
    pylab.close(1)
    fig = pylab.figure(1,figsize=tuple(reversed([(float(i)/dpi)*scale for i in m.shape])),dpi=dpi)
    ax = pylab.matshow(m,fignum=1)
    ax = pylab.figure(1).axes[0]
    if len(polys) > 0:
        for c,p in polys:
            ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c,lw=1))
    if len(hatchpolys) > 0:
        for c,h,p in hatchpolys:
            ax.add_patch(matplotlib.patches.Polygon(p,fc='none',ec=c,lw=1,hatch=h))
    if len(lines) > 0:
        for c,(X,Y) in lines:
            pylab.plot(X,Y,c)
    if len(bars) > 0:
        for c, (X,Y,bottom,kwargs) in bars:
            pylab.bar(X,Y,color=c,edgecolor=c,bottom=bottom,**kwargs)
    pylab.xticks([])
    pylab.yticks([])
    pylab.plot()
    pylab.ylim(m.shape[0],0)
    pylab.xlim(0,m.shape[1])
    pylab.subplots_adjust(0.01,0.01,0.99,0.99)
    
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    pi = Image.open(buf).convert('RGB')
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pi.tostring(),pi.size[0]*3)
    buf.close()
    return cv_im


def init_frames(stream,seg_frame_num=1800):
    frames = []
    for i in xrange(seg_frame_num):
        frames.append(array_from_stream(stream,normed=True,img_smooth_kernel=SMOOTH5))
    currsum = reduce(lambda x, y: x+y, frames)
    denom = numpy.ndarray(frames[0].shape,dtype=float)
    denom[:,:] = seg_frame_num
    return frames,currsum,denom

def shift_frames_return_diff(stream,frames,currsum,denom,seg_frame_num=1800,transform=''):
    f0 = frames.pop(0)
    frames.append(array_from_stream(stream,normed=True,img_smooth_kernel=SMOOTH5))
    currsum -= f0
    currsum += frames[-1]
    frameav = currsum/denom
    mdiff = frames[int(seg_frame_num/2)]-frameav
    if transform:
        if transform == 'invert':
            mdiff = -1 * mdiff
        elif transform == 'absval':
            mdiff = numpy.abs(mdiff)
        else:
            errstr = 'transform can only be ["invert","absval"], supplied: %s' % transform
            raise ValueError, errstr
    return mdiff

def seek_in_stream(stream,seek_frame_num):
    for i in xrange(seek_frame_num):
        null = cv.QueryFrame(stream)

###################################
# PARAMETER FITTING
###################################

def analyze_pass1(vid,seglen,nframes,step,nreps,target_int_step,start_offset=0):
    from video_analysis import mousezopt
    scores_dicts = []
    print >> sys.stderr, 'load first pass scores',
    for i in range(nreps):
        offset = (i*seglen*step)+start_offset
        scores_dicts.append(eval(open(mousezopt.generate_outfile_names(vid,seglen,offset,nframes,None,None,None)[0]).read()))
        print >> sys.stderr, '.',
    print >> sys.stderr, 'done'

    pass1_maxint = max([max(d.keys()) for d in scores_dicts])

    print >> sys.stderr, 'interpolate score vectors',
    scores_vectors = []
    for score_d in scores_dicts:
        print >> sys.stderr, '.',
        v = [None]*int((pass1_maxint/target_int_step) + 1)
        x,y = Util.dezip(sorted(score_d.items()))
        for xi,yi in score_d.items():
            v[int(xi/target_int_step)-1] = yi
        idx0 = v.index(0)
        v[idx0] = 1
        v_interp = Util.smooth(v,3,interpolate_nones=True)
        for i in range(idx0):
            v_interp[i] = 0
        scores_vectors.append(numpy.array(v_interp))
    print >> sys.stderr, 'done'

    return scores_vectors
    

def run_mousezopt(vid,seglen=900,nframes=300,step=2,nreps=60,target_int_step=0.001,n_obj=1,pass1queue = 'short_serial',transform='',start_offset=0):
    '''needs docstring: esp 2+ object tracking (test first)

    NB: start_offset is in frames
    '''
    from video_analysis import mousezopt,mousezopt_summary_helper
    
    def generate_cmds(peak_max=None,peak_min=None,target_int_step=None):
        cmds = []
        for i in range(nreps):
            offset = (i*seglen*step)+start_offset
            outfiles = mousezopt.generate_outfile_names(vid,seglen,offset,nframes,peak_max,peak_min,target_int_step)
            if all([(os.path.exists(outf) and os.path.getsize(outf) > 0) for outf in outfiles]):
                pass
            else:
                if peak_max is None:
                    cmds.append('mousezopt.py %s %s %s %s %s \'"%s"\'' % (vid,seglen,offset,nframes,n_obj,transform))
                else:
                    cmds.append('mousezopt.py %s %s %s %s %s \'"%s"\' %s %s %s' % (vid,seglen,offset,nframes,n_obj,transform,peak_max,peak_min,target_int_step))
        return cmds		

    def generate_helper_cmds():
        cmds = []
        for cutoff in numpy.arange(peak_max,peak_min,-target_int_step):
            outfiles = mousezopt_summary_helper.generate_outfile_names(vid,seglen,step,nframes,nreps,target_int_step,cutoff)
            if all([(os.path.exists(outf) and os.path.getsize(outf) > 0) for outf in outfiles]):
                pass
            else:
                cmds.append('mousezopt_summary_helper.py %s %s %s %s %s %s %s %s %s %s %s' % (vid, seglen, step, nframes, nreps, start_offset, cutoff, peak_max, peak_min, target_int_step, n_obj))
        return cmds

    vidbase,idx,analysis_win = os.path.splitext(vid)[0].rsplit('_',2)
    mousezopt_out = os.path.join(os.path.dirname(vidbase),idx,analysis_win,'mousezopt')

    #RUN PASS 1
    try:
        os.makedirs(mousezopt_out)
    except:
        pass

    
    cmds = generate_cmds()
    if cmds:
        logfile = os.path.join(mousezopt_out,'seg%s_size%s-pass1.log' % (seglen,nframes))
    
        jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,pass1queue,jobname_base='mzopt%sfr' % seglen)
        LSF.lsf_wait_for_jobs(jobids,logfile,pass1queue,namedict=namedict,kill_if_all_ssusp=True)
        
        time.sleep(10)
        
        cmds = generate_cmds()
        while len(cmds) > 0:
            jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,pass1queue,jobname_base='mzopt%sfr' % seglen)
            LSF.lsf_wait_for_jobs(jobids,logfile,pass1queue,namedict=namedict,kill_if_all_ssusp=True)
            
            cmds = generate_cmds()

    #ANALYZE PASS 1
    scores_vectors = analyze_pass1(vid,seglen,nframes,step,nreps,target_int_step,start_offset=start_offset)

    #FIND BEST CANDIDATES
    best_cutoffs = sorted(zip(reduce(lambda x,y:x+y, scores_vectors),numpy.arange(len(scores_vectors[0]))*target_int_step))[-50:]
    peak_min = min([c for s,c in best_cutoffs])
    peak_max = max([c for s,c in best_cutoffs])
    while peak_max - peak_min > 50 * target_int_step:
        best_cutoffs = best_cutoffs[1:]
        peak_min = min([c for s,c in best_cutoffs])
        peak_max = max([c for s,c in best_cutoffs])

    print >> sys.stderr, 'pass 2 tests %s - %s' % (peak_min,peak_max)
    
    #RUN PASS 2
    cmds = generate_cmds(peak_max,peak_min,target_int_step)
    if cmds:
        logfile = os.path.join(mousezopt_out,'seg%s_size%s-pass2.log' % (seglen,nframes))
    
        jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='mzopt%sfr' % seglen)
        LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict,kill_if_all_ssusp=True)
        
        time.sleep(10)
        
        cmds = generate_cmds(peak_max,peak_min,target_int_step)
        while len(cmds) > 0:
            jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='mzopt%sfr' % seglen)
            LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict,kill_if_all_ssusp=True)
            
            cmds = generate_cmds(peak_max,peak_min,target_int_step)

    #ANALYZE PASS 2
    # (now parallel; see below)
    '''
    ols_dicts = []
    print >> sys.stderr, 'load second pass blob outlines',
    for i in range(nreps):
        offset = i*seglen*step
        ols_dicts.append(eval(open(mousezopt.generate_outfile_names(vid,seglen,offset,nframes,peak_max,peak_min,target_int_step)[1]).read()))
        print >> sys.stderr, '.',
    print >> sys.stderr, 'done'

    #get a dummy frame for pixel dimensions
    stream = cv.CaptureFromFile(vid)
    SHAPE = array_from_stream(stream).shape

    pass2_scores_dict = {}
    pass2_scoring_dists_dict = {}
    for cutoff in numpy.arange(peak_max,peak_min,-target_int_step):
        print >> sys.stderr, 'concatenate blob outlines at cutoff %s ...' % cutoff,
        ols_cat = reduce(lambda x,y: x+[[]]+y, [([v for k,v in ols_d.items() if numpy.abs(k-cutoff)<target_int_step/2]+[[]])[0] for ols_d in ols_dicts])
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'find object arcs ...' ,
        objs = find_objs(ols_cat,SHAPE)[0]
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'calculate scoring distributions ...' ,
        size_h,size_bins,fol_h,fol_bins = get_object_arc_param_dists(ols_cat,SHAPE,size_binw=10,fol_binw=0.01)
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'score objects ...' ,        
        keep,drop = greedy_objs_filter(objs,ols_cat,size_h,size_bins,fol_h,fol_bins,SHAPE)
        sscore = sum([score_object_arc_size(o,ols_cat,size_h,size_bins) for o in keep])
        fscore = sum([score_object_arc_fol(o,ols_cat,fol_h,fol_bins,SHAPE) for o in keep])
        print >> sys.stderr, 'done'
        print >> sys.stderr, 'cutoff %s size: %s fol: %s score: %s' % (cutoff, sscore,fscore,sscore+fscore)
        pass2_scores_dict[cutoff] = sscore+fscore
        pass2_scoring_dists_dict[cutoff] = (size_h,size_bins,fol_h,fol_bins)

    '''
    print >> sys.stderr, 'run per-cutoff global scoring'
    cmds = generate_helper_cmds()
    if cmds:
        logfile = os.path.join(mousezopt_out,'seg%s_size%s-helper.log' % (seglen,nframes))
    
        jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='mzopt-help%sfr' % seglen,bsub_flags='-R "select[mem > 30000]"')
        LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict,kill_if_all_ssusp=True)
        
        time.sleep(10)
        
        cmds = generate_helper_cmds()
        while len(cmds) > 0:
            jobids,namedict = LSF.lsf_jobs_submit(cmds,logfile,'normal_serial',jobname_base='mzopt-help%sfr' % seglen,bsub_flags='-R "select[mem > 30000]"')
            LSF.lsf_wait_for_jobs(jobids,logfile,'normal_serial',namedict=namedict,kill_if_all_ssusp=True)
            
            cmds = generate_helper_cmds()

    print >> sys.stderr, 'collate global scores'
    pass2_scores_dict = {}
    pass2_scoring_dict = {}
    for cutoff in numpy.arange(peak_max,peak_min,-target_int_step):
        score_fname, model_fname = mousezopt_summary_helper.generate_outfile_names(vid,seglen,step,nframes,nreps,target_int_step,cutoff)
        pass2_scores_dict.update(eval(open(score_fname).read()))
        pass2_scoring_dict.update(eval(open(model_fname).read()))
    
            
    return pass2_scores_dict,pass2_scoring_dict
        
        
    #20120129 - OLD
    '''
    opt_counts = defaultdict(list)
    opt_sizes = defaultdict(list)
    match_counts = defaultdict(list)
    for f in glob(os.path.join(mousezopt_out,'seg%s_size%s*.tab' % (seglen,nframes))):
        d = [tuple([float(c) for c in l.strip().split()]) for l in open(f)]
        for cutoff,matchcount,bestcount,size in d:
            match_counts[cutoff].append(matchcount)
            opt_counts[cutoff].append(bestcount)
            opt_sizes[cutoff].append(size)
    return match_counts,opt_counts,opt_sizes
    '''    

def choose_cutoff(scores_dict,cut_step):
    d_sort = sorted(scores_dict.items(),key=lambda x:x[1],reverse=True)
    last = numpy.inf
    for i,(cutoff,score) in enumerate(d_sort):
        #print last,score,cutoff,i
        if numpy.abs(last-cutoff) <= 2*cut_step:
            return i-1,d_sort[i-1][0]
        last = cutoff
    return (-1,None)


###################################
# VIDEO/FILE OPERATIONS
###################################

def vid_duration(video):
    t = re.search('Duration: (\d+?):(\d+?):(\d+?)\.',subprocess.Popen('ffmpeg -i '+video,stderr=subprocess.PIPE,shell=True).stderr.read()).groups()
    return (int(t[0])*60*60) + (int(t[1])*60) + int(t[2])

def diff_first_and_last_frames(vid):
    vidlen = vid_duration(vid)
    out = os.path.splitext(vid)[0]
    os.system('vid2png.py %s %s 0 1 1' % (vid,out))
    os.system('vid2png.py %s %s %s 1 1' % (vid,out,vidlen-1))
    m1 = numpy.asarray(Image.open(sorted(glob(out+'/*.png'))[0]).convert('L'))
    m2 = numpy.asarray(Image.open(sorted(glob(out+'/*.png'))[-1]).convert('L'))
    return(m1.mean()-m2.mean())

def diff_first_and_last_frames_on_dir(d,ext='MPG'):
    '''convenience function to run diff_first_and_last_frames over the contents of a directory <d>'''
    vids = sorted(glob(d+'/*.'+ext))
    return [(v,diff_first_and_last_frames(v)) for v in vids]

def parallel_cidiff(d,ext='.MPG'):
    vids = sorted(glob(d+'/*'+ext))
    cmds = ['calc_intensity_diff.py '+v for v in vids]
    jobids,namedict = LSF.lsf_jobs_submit(cmds,d+'/log','short_serial',jobname_base='calc_intensity_diff')
    LSF.lsf_wait_for_jobs(jobids,d+'/restart-log','short_serial',namedict=namedict)
    diffs = sorted(glob(d+'/*.diff'))
    if len(diffs) != len(vids):
        raise ValueError, 'diffs (%s) and vids (%s) must be same len (%s, %s respectively)' % (diffs, vids, len(diffs), len(vids))
    else:
        diffvals = numpy.array([float(open(f).read()) for f in diffs])
    return vids[diffvals.argmax()],vids[diffvals.argmin()]

def peak_intensty_change(vid,num_jobs=40,summary_funct=numpy.mean,change_funct=max):
    '''given a video filename returns the first second which satisfies <change_funct> (i.e. whose value is returned by) 
    among all 1 second frames (un-normalized) transformed by <summary_funct>'''
    
    imdir = parallel_v2p(vid,1,num_jobs=num_jobs)
    ims = sorted(glob(imdir+'/*.png'))
    ars = numpy.array([summary_funct((numpy.asarray(Image.open(im).convert(mode='L')))) for im in ims])
    diffs = ars[:-1] - ars[1:]
    val = change_funct(diffs)
    return list(diffs).index(val)

def calculate_dusk_till_dawn(d,dark_pad=5,ext='.MPG'):
    vids = sorted(glob(d+'/*'+ext))
    first,last = parallel_cidiff(d)
    startsec = peak_intensty_change(first,change_funct=max)+dark_pad
    endsec = peak_intensty_change(last,change_funct=min)-dark_pad
    return first,last,startsec,endsec

def generate_dusk_till_dawn(d,final,ext='.MPG',dark_pad=5):
    '''generates a single video <final> with same encoding as inputs spanning lights-out to lights-on
    probably no longer the desired functionality - probably looking for v2p_dusk_till_dawn'''
    first,last,startsec,endsec = calculate_dusk_till_dawn(d,dark_pad)
    vids = sorted(glob(d+'/*'+ext))
    ifirst = vids.index(first)
    ilast = vids.index(last)
    invids = vids[ifirst+1:ilast]
    print >> sys.stderr, '%s @ %s to %s at %s including %s' % (first,startsec,last,endsec,invids)
    os.system('ffmpeg -ss %s -i %s -vcodec copy -acodec copy -sameq %s/first%s' % (startsec,first,d,ext))
    os.system('ffmpeg -t %s -i %s -vcodec copy -acodec copy -sameq %s/last%s' % (endsec,last,d,ext))
    os.system('cat %s/first%s %s %s/last%s > %s' % (d,ext,' '.join(invids),d,ext,final))

def v2p_dusk_till_dawn(d,fps,num_jobs=60,tdir=None,dark_pad=5,start_end=None,ext='.MPG',restart_z=12,return_start_end=False):
    '''given a dir of videos (assumes consecutive) generates a dir <d>/duskTillDawn/<fps>fps/
    containing consecutively labeled pngs from onset of dark to lights-on
    
    the times and videos to use as start and stop can be provided as <start_end> in the form:
    (first,last,startsec,endsec)
    
    otherwise calculate_dusk_till_dawn is called for these values
    unless restart_z=None, will invoke restarting long-running jobs per LSF.lsf_wait_for_jobs
    if return_start_end=True, cuts out after calculation step (doesn't actually run frame extraction')
    '''
    
    if tdir is None:
        tdir = os.path.join(d,'duskTillDawn/%sfps/' % fps)
        
    if start_end is None:
        first,last,startsec,endsec = calculate_dusk_till_dawn(d,dark_pad=dark_pad)
    else:
        first,last,startsec,endsec = start_end
        
    vids = sorted(glob(d+'/*'+ext))
    ifirst = vids.index(first)
    ilast = vids.index(last)
    invids = vids[ifirst+1:ilast]
    tot_in = sum([vid_duration(v) for v in invids])
    tot_sec = (vid_duration(first) - startsec) + endsec + tot_in
    tot_hours = tot_sec/3600.0
    print >> sys.stderr, 'from %s @ %s to %s @ %s: %s sec (%0.2f hr)\nextracting frames...' % (first,startsec,last,endsec,tot_sec,tot_hours)
    
    if return_start_end:
        return first,last,startsec,endsec
    
    #write bounds tuple to d+"/start_end.tuple
    open(d+'/start_end.tuple','w').write('("%s","%s",%s,%s)' % (first,last,startsec,endsec))
    
    #start with the truncated first vid
    parallel_v2p(first,fps,tdir,startsec,num_jobs=num_jobs,restart_z=restart_z)
    rename_images_from_zero(tdir)
    
    #proceed through each full vid in turn
    for i,v in enumerate(invids):
        print >> sys.stderr, 'extracting %s of %s' % (i+1,len(invids))
        foffset = len(glob(tdir+'/*.png'))
        parallel_v2p(v,fps,tdir,num_jobs=num_jobs,foffset=foffset,restart_z=restart_z)
    
    #then finish off the last partial
    foffset = len(glob(tdir+'/*.png'))
    parallel_v2p(last,fps,tdir,global_end=endsec,num_jobs=num_jobs,foffset=foffset,restart_z=restart_z)
    print >> sys.stderr, 'test files, drop failures:'
    os.system('drop_broken_pngs.py '+tdir)
    print >> sys.stderr, 're-zero:'
    rename_images_from_zero(tdir)
    return tdir

def parallel_p2v(pdir,fps, vid, dimensions, frames_per_seqment=44000, vidsize=4500, restart_z=12):
    '''splits a single large dir of pngs into many smaller dirs, makes individual MPGs for each split;
    merges MPGs and makes a final .mp4 targeted at about <vidsize> megabytes'''
    
    pass

def parallel_v2p(vid, fps, tdir=None, global_start=0, global_end=None, num_jobs=10,lock=True,foffset='',restart_z=24,queue='normal_serial',cropsdict=None,crops=None,extract_windows=None):
    '''launches parallelized (lsf) runs to split a video into stills
    
    cropsdict is a dict of crop tuples {mouse:(left,top,right,bottom),} or a file of same. if dict, writes cropsdict.dict file to vid dir.
    
    returns ids of running jobs if lock is False, else returns tdir (or in case of cropsdict, returns list of tdirs)
    '''
    
    if extract_windows is not None:
        return [parallel_v2p(vid,fps,tdir,s,e,num_jobs,lock,foffset,restart_z,queue,cropsdict,crops) for s,e in extract_windows]
    
    
    if not (global_start or global_end):
        win = 'all'
    elif not global_start and global_end:
        win = 'start-%05d' % int(global_end)
    elif global_start and not global_end:
        win = '%05d-end' % int(global_start)
    elif global_start and global_end:
        win = '%05d-%05d' % (int(global_start),int(global_end))
    
    if cropsdict is not None:
        if isinstance(cropsdict,str):
            cropsdict = eval(open(cropsdict).read())
        else:
            cdfile = os.path.join(os.path.dirname(vid),'cropsdict.dict')
            open(cdfile,'w').write(cropsdict.__repr__())
        #outbase = vid.rsplit('/',2)[0]+'/%%s/%s/%sfps/' % (win,fps) removed with auto-append below
        outbase = vid.rsplit('/',2)[0]+'/%s/'
        return [parallel_v2p(vid,fps,outbase%k,global_start,global_end,num_jobs,lock,foffset,restart_z,queue,crops=v) for k,v in cropsdict.items()]
	

    if tdir is None:
        tdir = vid.rsplit('.',1)[0]+'/%s/%sfps/png/' % (win,fps)
    else:
        tdir = os.path.join(tdir,'%s/%sfps/png/' % (win,fps))
        
    if global_end is None:
        global_end = vid_duration(vid)
        
    tot = global_end - global_start
    
    try:
        os.makedirs(tdir)
    except OSError:
        pass
    
    v2p = 'vid2png.py'
    
    step = tot/num_jobs
    
    if crops is not None:
        #changed for new ffmpeg
        #cropstr = '\\"-cropleft %s -croptop %s -cropright %s -cropbottom %s\\"' % crops
        cropstr = '\\"-vf crop=in_w-%s:in_h-%s:%s:%s\\"' % (crops[0]+crops[2],crops[1]+crops[3],crops[0],crops[1])
    else:
        cropstr = '.'
	
    cmds = []
    for i in range(global_start,global_end,step)[:-1]:
        cmds.append('%s %s %s %s %s %s %s %s' % (v2p, vid, tdir, i, step, fps, cropstr, foffset))
    if i+step < global_end:
        cmds.append('%s %s %s %s %s %s %s %s' % (v2p, vid, tdir, i+step, global_end - (i+step), fps, cropstr, foffset))
        
    if lock:
        while cmds:
            if restart_z is not None:
                jobids,namedict = LSF.lsf_jobs_submit(cmds,tdir+'log',queue,jobname_base='vid2png')
                LSF.lsf_wait_for_jobs(jobids,tdir+'log','hoekstra',namedict=namedict,restart_z=restart_z)
            else:
                jobids,namedict = LSF.lsf_jobs_submit(cmds,tdir+'log',queue,jobname_base='vid2png')
                LSF.lsf_wait_for_jobs(jobids,tdir+'log','normal_serial',namedict=namedict)
            time.sleep(10)
            cmds = [c.replace('"','\\"') for c in LSF.lsf_no_success_from_log(tdir+'log')]
            if cmds: print >> sys.stderr, 'rerunning %s' % cmds
        return tdir
    else:
        jobids,namedict = LSF.lsf_jobs_submit(cmds,tdir+'log',queue,jobname_base='vid2png')
        return jobids,tdir

def get_bad_images(imagedir,type='png',numprogs = 20):
    bad = []
    images = sorted(glob(imagedir+'/*.'+type))
    print >> sys.stderr, 'analyze %s images' % len(images)
    outstep = len(images)/numprogs
    for i,im in enumerate(images):
        if i%outstep == 0:
            print >> sys.stderr, 'processed %s / %s' % (i,len(images))
        try:
            imobj = Image.open(im).convert('L')
        except:
            print >> sys.stderr, 'image %s failed to load' % im
            bad.append(im)
    return bad

def rename_images_from_zero(imagedir,type='png',digits=7, zero=0, clear_below=None):
    images = sorted(glob(os.path.join(imagedir,'*.'+type)))
    if clear_below is not None:
        print >> sys.stderr, 'clear_below %s invoked; checking...' % clear_below,
        #empties = [f for f in images if os.path.getsize(f) <= clear_below]
        empties = get_bad_images(imagedir)
        if len(empties) != 0:
            print >> sys.stderr, '%s empty files present, clearing.' % len(empties)
            for f in empties:
                os.unlink(f)
            print >> sys.stderr, 'done, proceeding.'
            images = sorted(glob(os.path.join(imagedir,'*.'+type)))
        else:
            print >> sys.stderr, 'none found'
        
    fstr = '/%%0%dd.%%s' % digits
    for i,f in enumerate(images):
        newf = f.rsplit('/',1)[0]+fstr % (i+zero,type)
        shutil.move(f,newf)

def check_sequential_files(d,ext,step=1):
    '''given directory and extension, checks that all files are in numerical series, incrementing by <step>'''
    
    files = sorted(glob(d+'/*.'+ext))
    last = int(os.path.splitext(os.path.split(files[0])[1])[0])
    gaps = []
    for f in files[1:]:
        this = int(os.path.splitext(os.path.split(f)[1])[0])
        if last+step != this:
            gaps.append((f,this-last))
        last = this
    return gaps
	
def check_contiguous_files(d,ext):
    '''given a directory and an extension, checks to make sure files of the form:
    <d>/start-end.<ext> are sequential,
    i.e. that 0000000-0000900.whatever is followed by 0000900-0001800.whatever
    returned list is tuples, first element is filename AFTER the gap, second is size (in files) of the gap'''
    def get_start_end(s):
        start,end = os.path.split(s)[1].split('.')[0].split('-')
        return start,end
    
    files = sorted(glob(d+'/*.'+ext))
    
    start,last = get_start_end(files[0])
    step = int(last)-int(start)
    gaps = []
    for f in files[1:]:
        try:
            start,end = get_start_end(f)
        except ValueError:
            continue
        if last != start:
            gapsize = (int(start)-int(last))/step
            gaps.append((f,gapsize))
        last = end
    return gaps


def timestamp_from_path(imagepath,fps=None):
    '''returns the timestamp (float seconds) of an image, given a pathname'''
    
    if fps is None:
        match = re.search(r'\/([\d\.]+)fps\/',imagepath)
        if match:
            fps = float(match.groups()[0])
        else:
            raise ValueError, 'no fps spec in path',imagepath
    else:
        fps = float(fps)
    
    match = re.search(r'\/(\d+)\.[pngj]{3}',imagepath)
    framenum = int(match.groups()[0])
    
    return framenum/fps

def filename2win(filename,delim='-'):
    '''given a filename in /path/to/file/start-end.extension
    returns (start,end) tuple
    '''
    s,e = os.path.splitext(os.path.basename(filename))[0].split(delim)
    return (int(s),int(e))

def get_files_by_win(filenames,win,offset_fr=0):
    ''' given a list of files /path/to/file/start-end.extension
    returns list of files start<win[0]<end through start<win[1]<end
    if offset_fr is supplied, this is assumed to be the video frame number corresponding to frame 0 in filenames
    '''
    filenames.sort()
    si = None
    for i,f in enumerate(filenames):
        s,e = [fr+offset_fr for fr in filename2win(f)]
        if s <= win[0] < e:
            si = i
        if s < win[1] <= e:
            break
    return filenames[si:i+1]

def load_normed_arrays(filelist,pix_av_win=None,img_smooth_kernel=None):
    '''currently hacked to allow tarball:filename indexing, but first element must have this structure!'''
    nars = []
    if '.tar:' in filelist[0]: #assumes a : means file in a tarball; only checks the first element
        import tarfile
        tarf,tarim = filelist[0].split(':')
        print >> sys.stderr, 'filelist contains tar references (".tar:"), loading tarball %s' % tarf
        tarobj = tarfile.open(tarf)
        tar = True
    else:
        tar = False
    
    for i in filelist:
        if tar:
            tarf,tarim = i.split(':')
            if tarf != tarobj.name:
                print >> sys.stderr, 'new tarball referenced, opening %s' % tarf
                tarobj = tarfile.open(tarf)
            im = tarobj.extractfile(tarim)
        else:
            im = i
        try:
            if img_smooth_kernel is not None:
                ar = numpy.asarray(Image.open(im).convert(mode='L').filter(img_smooth_kernel))
            else:
                ar = numpy.asarray(Image.open(im).convert(mode='L'))
        except:
            print >> sys.stderr, 'failed opening %s' % im
            raise
        if ar.any():
            ar = Util.normalize(ar)
            if pix_av_win:
                nars.append(Util.smooth(ar, pix_av_win))
            else:
                nars.append(ar)
    return numpy.array(nars)

def extract_keyframe(vid,sec=60,**kwargs):
    '''given a video (and optionally which second to sample) returns a numpy array (b+w, normed) of a single frame.
    '''
    keypng = os.path.join(os.path.splitext(vid)[0] , 'key01.png')
    #if not os.path.exists(keypng):
    if 1:
        try:
            os.makedirs(os.path.splitext(vid)[0])
        except:
            pass
        execstr = 'ffmpeg -ss %s -t 1 -i %s -r 1 -y %s' % (sec,vid,os.path.join(os.path.splitext(vid)[0] , 'key%02d.png'))
        os.system(execstr)
    fr = load_normed_arrays([keypng],**kwargs)[0]
    return fr

def extract_change_over_vid(vid,pad_into=60):
    '''given a video and optionally the number of seconds in from both ends to compare
    
    returns a frame representing change over the duration of the video (2*f1-f2)
    '''
    
    fr = extract_keyframe(vid,pad_into) - 2.5 * extract_keyframe(vid, vid_duration(vid)-pad_into)
    return fr

def extract_change_over_pngs(tdir,pad_into=1800):
    '''given a dir of .png files as per v2p
    
    return a matrix of intensity differences of pixels from the pad_into and -pad_into files'''
    
    ims = sorted(glob(tdir+'*.png'))
    
    f1 = numpy.asarray(Image.open(ims[pad_into]).convert(mode='L'))
    f2 = numpy.asarray(Image.open(ims[-1*pad_into]).convert(mode='L'))
    
    return f2 - (f1-f2)

def average_frames(frames,pix_av_win=None,num_wins=1):
    '''returns a single average of a list of frames
    
    if frames are strings, will treat as filenames, loading with pixel averaging of pix_av_win'''
    
    if num_wins > 1:
        winlen = len(frames) / num_wins
        return [average_frames(frames[i:i+winlen],pix_av_win) for i in range(0,len(frames),winlen)]
    
    if isinstance(frames[0],numpy.ndarray):
        denom = numpy.ndarray(frames[0].shape,dtype=float)
        denom[:,:] = len(frames)
        return reduce(lambda x, y: x+y, frames) / denom
    elif isinstance(frames[0],str):
        first = load_normed_arrays(frames[:1],pix_av_win)[0]
        denom = numpy.ndarray(first.shape,dtype=float)
        denom[:,:] = len(frames)
        running_sum = first
        for f in frames[1:]:
            running_sum += load_normed_arrays([f],pix_av_win)[0]
        return running_sum / denom

def find_mouse(frame,background,zcut=6,origin=(0,0),abs_val=False,min_pixels_above_cut=1,outline_zcut=None,grow_by=0,preshrink=1):
    diff = Util.zscore(frame-background)
    #print len(diff[diff>zcut])
    if abs_val:
        diff = numpy.abs(diff)
    if diff.max() > zcut and len(diff[diff>zcut]) > min_pixels_above_cut:
        pix = diff.argmax()		
        if outline_zcut is not None:
            ol,term = chain_outlines_from_mask(diff>outline_zcut,grow_by=grow_by,preshrink=preshrink)
            ol = [p for p in ol if diff[mask_from_outline(p,frame.shape)].max() > zcut]
            if len(ol) > 0:
                ol.sort(key=size_of_polygon,reverse=True)
                
                if grow_by:
                    zsc = [diff[shrink_mask(mask_from_outline(p,frame.shape),grow_by)].mean() for p in ol]
                else:
                    zsc = [diff[mask_from_outline(p,frame.shape)].mean() for p in ol]
                for i,z in enumerate(zsc):
                    if z is numpy.nan:
                        zsc[i] = None
                loc = center_of_polygon(ol[0])
                return tuple(numpy.array(loc)-numpy.array(origin)),ol,zsc
            else:
                return None,[],[]
        else:
            loc = (pix % diff.shape[1], pix / diff.shape[1])
            return tuple(numpy.array(loc)-numpy.array(origin))
    else:
        if outline_zcut is not None:
            ol,term = chain_outlines_from_mask(diff>outline_zcut,grow_by=grow_by,preshrink=preshrink)
            ol = [p for p in ol if diff[mask_from_outline(p,frame.shape)].max() > zcut]				
            if len(ol) > 0:
                ol.sort(key=size_of_polygon,reverse=True)
                
                if grow_by:
                    zsc = [diff[shrink_mask(mask_from_outline(p,frame.shape),grow_by)].mean() for p in ol]
                else:
                    zsc = [diff[mask_from_outline(p,frame.shape)].mean() for p in ol]
                for i,z in enumerate(zsc):
                    if z is numpy.nan:
                        zsc[i] = None
                return None,ol,zsc
            else:
                return None,[],[]
        else:
            return None

def xy_from_idx(idx,shape):
    '''return (x,y) coords for a given index in flattened vector'''
    xy = (idx % shape[1], idx / shape[1])
    return xy

def pix_to_cm_factor(scale_points):
    '''returns a mean scaling factor f
    such that x*f=c where x is a linear distance in pixels and c is that dist in cm
    
    <scale_points> is a dict with keys ((x1,y1),(x2,y2))
    and values the real dist between the two in cm
    '''
    
    scales = []
    
    for points,dist in scale_points.items():
        pix = hypotenuse(*points)
        scales.append(dist/pix)
        
    scales = numpy.array(scales)
    
    print >> sys.stderr, 'scaling factors: %s\nmean: %s std: %s' % (scales,scales.mean(),scales.std())
    return scales.mean()

def find_ground(frame,mask=None,burrow_mask=None,ybounds=None,offset=5,smoothby=20,zcut=3):
    '''uses change in pixel intensity to find the ground, given masks for overall problem regions (i.e. support uprights)
    and burrow(s) i.e. most changed pixels
    
    try 
    >>> mask = fullav_smooth > 0.19
    >>> mask10 = vidtools.grow_mask(mask,10)
    guided by pixel intensity histogram
    
    and (to find differences of greater than 2 sigma over mean that are more than 20 pixels underground, and in the top 400 pixels of the frame)
    >>> burrow_mask = vidtools.find_burrow(images,mask10,change_zcut=2,depress_groundmask=20,ybounds=(0,400))
    >>> burrow_mask5 = vidtools.grow_mask(burrow_mask,5)'''
    
    if not ybounds:
        ybounds = (0,len(frame))
        
    ground = []
    
    if isinstance(mask,str):
        print >> sys.stderr, 'mask supplied as string %s, assuming filename, loading with shape %s' % (mask,frame.shape)
        mask = numpy.fromfile(mask,dtype=bool).reshape(frame.shape)
    
    for i in range(len(frame[0])):
        if mask is not None:
            if mask[ybounds[0]:ybounds[1],i].any():
                ground.append(None)
                continue
        win = frame[ybounds[0]:ybounds[1],i]
        diff = [0]*offset+list(win[offset:] - win[:-1*offset])
        if burrow_mask is not None:
            mask_win = burrow_mask[ybounds[0]:ybounds[1],i]
        else:
            mask_win = numpy.zeros(win.shape)
        diff = Util.zscore(Util.smooth(Util.subtract_mask(numpy.array(diff),mask_win,0),smoothby))
        infl = diff.argmax()
        z = diff.max()
        if z > zcut:
            ground.append(infl)
        else:
            ground.append(None)
        
    return numpy.array(ground).transpose()
	
def mean_diffs(vect):
    '''retuns a vector where each value is the difference of the means of the values before and after that index'''
    diffs = numpy.zeros(len(vect))
    for i in xrange(1,len(vect)):
        diffs[i] = numpy.mean(vect[i:]) - numpy.mean(vect[:i])
    return diffs

def best_split(vect,window=None):
    '''given a vector, find the index that maximizes the difference between vect[:i] and vect[i:].  
    If window (start,end) is supplied, return the best i in that window'''
    if window is None:
        start,end = (0,len(vect))
    else:
        start,end = window
    diffs = mean_diffs(vect[start:end])
    
    return diffs.argmax()+start

def find_ground3(frame,seed,xybounds=None,be=None,win=20):
    '''modify to start at left groundlevel, find max in window, return all
    <seed> is y coord of leftmost ground (x is xybound low x) '''
    
    flip = frame.transpose()
    
    if xybounds:
        [(tx,ty),(bx,by)] = xybounds
    else:
        [(tx,ty),(bx,by)] = [(0,0),flip.shape]
        
    #meandiffs = numpy.zeros(flip.shape,dtype=float)
    ground = []
    previ = seed
    for i,col in enumerate(flip[tx:bx]):
        #if burrow entrance given, skip the area around it
        if be is not None and (i+tx in range(be[0]-10,be[0]+10)):
            ground.append(None)
            continue
        
        start = max(ty,previ-win)
        end = min(by,previ+win)
        split = best_split(col,(start,end))
        ground.append(split)
        
        if split:
            previ = split
    
    #restore bounds
    ground = [None]*tx + ground + [None]*(len(flip)-bx)
    
    return ground #,meandiffs.transpose()

def find_ground2(frame,zcut=3,win=40,top=10,xybounds=None):
    [(tx,ty),(bx,by)] = xybounds
    flip = frame[ty:by,tx:bx].transpose()
    meandiffs = numpy.zeros(flip.shape,dtype=float)
    ground = []
    for i,col in enumerate(flip):
        for j,val in enumerate(col):
            start = max(0,j-win)
            end = min(len(col),j+win)
            above = max(0,col[start:j].mean())
            below = max(0,col[j:end].mean())
            meandiffs[i,j] = below - above
        #ground.append(meandiffs[i][top:].argmax() + top)
    
    for vect in meandiffs:
        zdiffs = Util.zscore(vect[top:])
        infl = zdiffs.argmax() + top
        z = zdiffs.max()
        if z > zcut:
            ground.append(infl+ty)
        else:
            ground.append(None)
        
    #restore bounds
    ground = [None]*tx + ground + [None]*(len(flip)-bx)

    return ground #,meandiffs.transpose()

def calc_split_mat_for_groundtrace(frame,offset):
    split = []
    for row in frame.transpose():
        ar = row[offset:] - row[:-offset]
        ar[row[offset:] == 0] = 0
        ar[row[:-offset] == 0] = 0
        numpy.array([0]*offset + list(ar))
        split.append(ar)
    return numpy.array(split).transpose()

def calc_diff_mat(frame,offset):
    dm = numpy.zeros(frame.shape,dtype=float)
    for i in xrange(offset,dm.shape[0]-offset):
        for j in xrange(offset,dm.shape[1]-offset):
            perim = []
            perim.extend(frame[i-offset,j-offset+1:j+offset])
            perim.extend(frame[i+offset,j-offset+1:j+offset])
            perim.extend(frame[i-offset+1:i+offset,j-offset])
            perim.extend(frame[i-offset+1:i+offset,j+offset])
            dm[i,j] = frame[i,j] - numpy.mean(perim)
    return dm

def rec_max(vec,pos,maxstep=None,_start=None,_step=0):
    maxpos = numpy.argmax(vec[pos-1:pos+2])
    if maxstep is not None:
        if _step == 0:
            _start = pos
        _step += 1
    if maxstep is not None and _step>maxstep:
        return _start
    if maxpos == 1:
        return pos
    else:
        return rec_max(vec,pos-1+maxpos,maxstep,_start,_step)

def find_ground_over_frames_matrix(frames,start_ground_anchors,end_ground_anchors,ahead_window=10,ahead_avg=5,fix_deviation=0.5,allow_passes=100,max_step_limit=None):
    '''20130326 BEST OPTION
    given frames (segment averages) as either list of 2D arrays (coverts to 3D array)
    or 3D array, and anchor points for first and last grounds
    ("ground_anchors" and "end_ground_anchors" in antfarm config)

    returns "surface" (list of lists of ground Y coords through time) of best ground
    '''
    from copy import deepcopy
    
    if type(frames) == list:
        frames = numpy.array(frames)
    print >> sys.stderr, 'calculate derivatives'
    frames_deriv = frames[:,1:,:] - frames[:,:-1,:]
    print >> sys.stderr, 'Z transform'
    frames_deriv_Z = numpy.array([Util.zscore(frames_deriv[:,:,x]) for x in xrange(frames_deriv.shape[2])]).transpose((1,2,0))
    ld = dict(start_ground_anchors)
    l = [ld.get(i,None) for i in xrange(frames[0].shape[1])]
    fit_l = [rec_max(frames_deriv_Z[0][:,x],anchor,max_step_limit) for x,anchor in enumerate(Util.smooth(l,5,interpolate_nones=True))]
    fit_surface = [fit_l]
    print >> sys.stderr, 'trace 1'
    for m in frames_deriv_Z[1:]:
        fit_l = [rec_max(m[:,x],anchor,max_step_limit) for x,anchor in enumerate(fit_surface[-1])]
        fit_surface.append(fit_l)

    #return fit_surface

    eld = dict(end_ground_anchors)
    el = [eld.get(i,None) for i in range(frames[0].shape[1])]
    end_fit_l = [rec_max(frames_deriv_Z[-1][:,x],anchor,max_step_limit) for x,anchor in enumerate(Util.smooth(el,5,interpolate_nones=True))]
    match_x = [x for x,y in enumerate(end_fit_l) if y == fit_surface[-1][x]]

    #return fit_surface,match_x

    filt_fit_surface = []
    print >> sys.stderr, 'trace 2'
    for iter,(fit_l,m) in enumerate(zip(fit_surface,frames_deriv_Z)):
        this_ld = dict([(x,fit_l[x]) for x in match_x])
        this_l = [this_ld.get(i,None) for i in xrange(frames[0].shape[1])]

        #filt_fit_l = Util.smooth(this_l,5,interpolate_nones=True)
        filt_fit_l = [rec_max(m[:,x],anchor,max_step_limit) for x,anchor in enumerate(Util.smooth(this_l,5,interpolate_nones=True))]

        filt_fit_surface.append(filt_fit_l)
        print >> sys.stderr, '\r %s/%s' % (iter+1,len(fit_surface)),

    #return filt_fit_surface

    new_gs = filt_fit_surface
    redo = True
    passes = 0
    print >> sys.stderr, '\nlocal fix'
    while redo:
        redo = False
        old_gs = new_gs
        new_gs = []
        for g in old_gs:
            new_gs.append([g[0]])
            for i,p in enumerate(g[1:-1],1):
                ev = ((g[i+1]-g[i-1])/2.0)+g[i-1]
                if -fix_deviation < p-ev < fix_deviation:
                    new_gs[-1].append(p)
                else:
                    new_gs[-1].append(ev)
                    redo = True
            new_gs[-1].append(g[-1])
        passes += 1
        print >> sys.stderr, '\r %s' % passes,
        if passes >= allow_passes:
            break

    print >> sys.stderr, '\nfuture fix'
    gs_futurefix = deepcopy(new_gs)
    for x in range(len(new_gs[0])):
        for t in range(1,len(new_gs)-ahead_window-ahead_avg):
            future_val = numpy.mean([new_gs[ft][x] for ft in range(t+ahead_window,t+ahead_window+ahead_avg)])
            if (new_gs[t][x] - future_val)**2 > (new_gs[t-1][x] - future_val)**2:
                gs_futurefix[t][x] = new_gs[t-1][x]
    print >> sys.stderr, '\nrevoke ground lowering'
    for t in range(1,len(gs_futurefix)):
        for x in range(len(gs_futurefix[t])):
            if gs_futurefix[t][x] > gs_futurefix[t-1][x]:
                gs_futurefix[t][x] = gs_futurefix[t-1][x]

    print >> sys.stderr, 'DONE'
    return gs_futurefix

def find_ground4(frame,groundpts,recursion=5,be=None,xybounds=None):
    '''given a list of ground points (can be ground_start from both ends) and an intensity matrix
    returns a ground array calculated as the best of a set of maximum_value_path computes
    based on mean_diff matrix'''
    
    def shift_off_zero(pt):
        if pt[0] == pt[1] == 0:
            return (1,1)
        elif pt[0] == 0:
            return (1,pt[1])
        elif pt[1] == 0:
            return (pt[0],1)
    
    candidate_grounds = {}
    candidate_groundlines = {}
    
    for offset in [2,5,10]:
        split = calc_split_mat_for_groundtrace(frame,offset)
        #gpfunc = lambda pt: (pt[0],pt[1]+offset)
        gpfunc = lambda pt: pt
        #gpfunc = shift_off_zero
        for win in [5,10,20]:
            #print >> sys.stderr, '%s-%s_f' % (offset,win)
            try:
                thesepaths = []
                for i, gp in enumerate(groundpts[:-1]):
                    thesepaths.extend(maximum_value_path(Util.smooth(split*100,win),gpfunc(gp),gpfunc(groundpts[i+1]),recursion)[1])
                candidate_grounds['%s-%s_f' % (offset,win)] = thesepaths
            except:
                print >> sys.stderr, '%s-%s_f failed' % (offset,win)
            #print >> sys.stderr, '%s-%s_r' % (offset,win)
            
            try:
                thesepaths = []
                for i, gp in enumerate(groundpts[:-1]):
                    thesepaths.extend(maximum_value_path(Util.smooth(split*100,win),gpfunc(groundpts[i+1]),gpfunc(gp),recursion)[1])
                candidate_grounds['%s-%s_r' % (offset,win)] = thesepaths
            except:
                print >> sys.stderr, '%s-%s_r failed' % (offset,win)

       
    for k,v in sorted(candidate_grounds.items()):
        if len(v) == 0:
            continue
        #print >> sys.stderr, 'finish line %s, len %s, unique x count: %s' % (k,len(v), len(set([x for x,y in v])))
        g = numpy.zeros(len(frame[0]))
        #print >> sys.stderr, 'for %s,offset %s' % (k,offset)
        for x,y in v:
            g[x] = y
        #g[g==0] = None
		
        x,y = [numpy.array(ar) for ar in Util.dezip(v)]
        g[:x.min()] = y[x.argmin()]
        g[x.max():] = y[x.argmax()]
        candidate_groundlines[k] = Util.smooth([gp > 0 and gp or None for gp in g],10,interpolate_nones=True) #+ 5  # +5 is HAX
    
    #add find_ground3 result to candidates list
    #FIX ME!
    #ground3 = find_ground3(frame,groundpts[0][1],xybounds=xybounds,be=be)
    #ground3[:groundpts[0][0]] = groundpts[0][1]
    #ground3[groundpts[-1][0]:] = groundpts[-1][1]
    #candidate_groundlines['0-find_ground3'] = Util.smooth(ground3,10,interpolate_nones=True)
    
    ratio,key = max([(evaluate_ground(frame,v),k) for k,v in candidate_groundlines.items()])
    
    print >> sys.stderr, 'selected %s, ratio %s' % (key, ratio)
    offset = int(key.split('-')[0])
    
    return candidate_groundlines[key]# - offset #, candidate_groundlines

def evaluate_ground(frame,ground):
    '''given an intensity matrix and a ground vector
    returns the ratio of mean intensities below to above'''
    
    above = []
    below = []
    for i,row in enumerate(frame.transpose()):
        abo = row[:ground[i]]
        above.extend(abo[abo!=0])
        bel = row[ground[i]:]
        below.extend(bel[bel!=0])
    
    return numpy.array(below).mean() / numpy.array(above).mean()

def best_ground(frame,grounds):
    '''given a frame intensity matrix and a list of ground vectors
    returns the index of the "best" ground, e.g. the ground vector that maximizes the difference between earth and sky'''
    '''
    flip = frame.transpose()
    
    bestdif = None
    bestidx = None
    for i,g in enumerate(grounds):
    thisdif = 0
    for x, vect in enumerate(flip):
    thisdif += vect[g[x]:].mean() - vect[:g[x]].mean()
    if thisdif > bestdif:
    bestdif,bestidx = thisdif,i
    return bestidx
    '''
    return numpy.array([evaluate_ground(frame,g) for g in grounds]).argmax()

def find_burrow(images,mask=None,change_zcut=3,smoothby=5,depress_groundmask=10,offsets=None,pct_to_average=0.01,frames_to_average=100,**ground_args):
    if offsets is None:
        pct = int(len(images)*pct_to_average)
        step = pct/frames_to_average
        if step == 0:
            step = 1
        offsets = ((pct,pct*2,step),(-2*pct,-1*pct,step))
    
    first = Util.smooth(average_frames(images[offsets[0][0]:offsets[0][1]:offsets[0][2]]),smoothby)
    last = Util.smooth(average_frames(images[offsets[1][0]:offsets[1][1]:offsets[1][2]]),smoothby)
    
    if isinstance(mask,str):
        print >> sys.stderr, 'mask supplied as string %s, assuming filename, loading with shape %s' % (mask,first.shape)
        mask = numpy.fromfile(mask,dtype=bool).reshape(first.shape)
    
    groundmask = mask_from_vector(Util.smooth(find_ground(first,mask,**ground_args),10,interpolate_nones=True)+depress_groundmask,first.shape)
    
    change = first - last
    change = Util.zscore(change)
    if mask is not None:
        change = Util.subtract_mask(change,mask,0.0)
    change = Util.subtract_mask(change,groundmask,0.0)
    
    return change > change_zcut

def classify_mouse(mouse,ground,activitymask):
    '''classify a mouse in the following scheme:
    
    0: mouse not found
    1: above ground, no activity (e.g. not digging)
    2: above ground, activity
    3: below ground, no activity
    4: below ground, activity
    given a mouse coord (tuple or None), a ground vector and a mask of recent activity'''
    
    if mouse is None:
        return 0
    
    if activitymask[mouse[1],mouse[0]]:
        actmod = 1
    else:
        actmod = 0
    
    return ((mouse[1] > ground[mouse[0]]) * 2) + actmod + 1

#exits given entry for maximum_value_path
exits = {0:(5,7,8),
	 1:(8,6,7),
	 2:(3,7,6),
	 3:(2,8,5),
	 5:(6,0,3),
	 6:(1,5,2),
	 7:(0,2,1),
	 8:(3,1,0)
	 }
#coords given exit IN (X,Y)!!! (not y,x as per actual matrix)
exit_coords = {1:lambda t:(t[0],t[1]-1),
	       2:lambda t:(t[0]+1,t[1]-1),
	       5:lambda t:(t[0]+1,t[1]),
	       8:lambda t:(t[0]+1,t[1]+1),
	       7:lambda t:(t[0],t[1]+1),
	       6:lambda t:(t[0]-1,t[1]+1),
	       3:lambda t:(t[0]-1,t[1]),
	       0:lambda t:(t[0]-1,t[1]-1),
	       }

def calc_entry_for_path(mat,point):
    '''given a starting point (maybe closest point to burrow entrance? :)
    returns the "entry" angle (see maximum_value_path docstr) and the start point on that angle
    '''
    
    entry = mat[point[1]-1:point[1]+2, point[0]-1:point[0]+2].argmax()
    startpoint = exit_coords[entry](point)
    entryfrom = exits[entry][-1]
    return entryfrom,startpoint

def maximum_value_path(mat,startpoint,endpoint,horizon=10,prox_to_end=10,kill=1000):
    '''defines a heuristic maximum value path through a 2D array via limited path recursion
    
    relative coordinates per <entry> as follows:
    
    [[0,1,2],
     [3, ,5],
     [6,7,8]]
    
     exit only allowed in 3 "forward" directions (e.g. entry=7, next points=(2,3,4)
     
     returns (score,path,done,lastentry)
     '''
    
    def extend_path(mat,entry,this_point,endpoint,prev_path,this_path,this_score,horizon,prox_to_end):
        #print >> sys.stderr, entry,this_point,len(this_path)
        
        new_path = this_path+[this_point]
        closer_by = hypotenuse(prev_path[-1],endpoint) - hypotenuse(this_point,endpoint)
        #closer_by=1
        new_score = this_score+(mat[this_point[1],this_point[0]] + closer_by)
        
        if this_point in prev_path+this_path or mat[this_point[1],this_point[0]] == 0:
            #print >> sys.stderr, 'dead end'
            return -numpy.infty,this_path,False,None
        elif len(this_path) == horizon:
            #print >> sys.stderr, 'reached horizon at %s (entry: %s), path %s score %s' % (this_point,entry,len(new_path),new_score)
            return new_score,new_path,False,entry
        elif hypotenuse(this_point,endpoint) < prox_to_end:
            #print >> sys.stderr, 'distance to end from %s (entry: %s): %s, path %s with score %s' % (this_point,entry,hypotenuse(this_point,endpoint),len(new_path),new_score)
            return numpy.infty,new_path,True,entry
        else:
            next_score,next_path,done,lastentry =  max([extend_path(mat,exits[e][-1],exit_coords[e](this_point),endpoint,prev_path,new_path,new_score,horizon,prox_to_end) for e in exits[entry]])
            return next_score,next_path,done,lastentry
    
    entry, curr_endpoint = calc_entry_for_path(mat,startpoint)
    done = False
    path=[startpoint]
    score=0
    
    while not done:
        if len(path) > kill:
            return None
        else:
            #print >> sys.stderr, 'at %s, entry on %s, path %s' % (curr_endpoint,entry,len(path))
            next_score,next_path,done,lastentry = max([extend_path(mat,exits[e][-1],exit_coords[e](curr_endpoint),endpoint,path,[],0,horizon,prox_to_end) for e in exits[entry]])
            if len(next_path) == 0 :
                done = True
            #print >> sys.stderr, 'selected path %s with score %s' % (next_path,next_score)
            path += next_path
            score += next_score
            curr_endpoint = path[-1]
            entry = lastentry
            
    #print >> sys.stderr, 'finished'
    return score,path

def find_burrow_over_frames_matrix(frames,mousemasks,grounds,suppress_ground,predug_poly=None,diff_min=0.2):
    '''first_segment_actouts takes contents of first segment burrow tracking from first-pass.
    if None, first segment will always be un-tracked (since i-1 and i+1 are required for burrow finding)

    diff_min is difference in average intensity to call burrow; 0.2 taken from summarize_segment_opencv hard-code
    (since mousemask is principally driving burrow calls, this is probably not crucial)
    '''
    SHAPE = frames[0].shape
    if predug_poly is None:
        pd_config_mask = numpy.zeros(SHAPE,dtype=bool)
    else:
        pd_config_mask = mask_from_outline(outline_from_polygon(predug_poly),SHAPE)

    if type(frames) == list:
        frames = numpy.array(frames)

    prevactmask = numpy.zeros(SHAPE,dtype=bool)
    pdmask = numpy.zeros(SHAPE,dtype=bool)
    groundmask = numpy.zeros(SHAPE,dtype=bool)
    prevactols = []
    newactols = []
    pdols = []
    
    print >> sys.stderr, 'track burrowing'
    for i,(fr,mm,g) in enumerate(zip(frames,mousemasks,grounds)):
        #add last new (if any) to previous
        if newactols and newactols[-1]:
                prevactmask += reduce(lambda x,y:x+y, [mask_from_outline(ol,SHAPE) for ol in newactols[-1]])

        if 1 < i < len(frames)-1:
            diffmat = frames[i-1] - frames[i+1]
        else:
            diffmat = numpy.zeros(SHAPE)

        groundmask[grow_mask(mask_from_vector(g,SHAPE),suppress_ground)] = True
        
        m = diffmat > 0.2
        m[mm] = True #add mousemask to burrow area
        m[grow_mask(shrink_mask(prevactmask,1),1)] = False #MASK PREVIOUS DIGGING
        m[pdmask] = False #AND PREDUG
        m[groundmask] = False #AND EVERYTHING ABOVE GROUND

        newols = chain_outlines_from_mask_shapely(m,preshrink=1,grow_by=1)
        prevols = chain_outlines_from_mask_shapely(prevactmask,preshrink=1,grow_by=1)

        newactols.append([])
        pdols.append([])
        prevactols.append(prevols)
        
        for ol in newols:
            try:
                olmask = mask_from_outline(ol,SHAPE)
                if any(olmask[pd_config_mask]):
                    pdmask[olmask] = True
                    pdols[-1].append(ol)
                else:
                    newactols[-1].append(ol)
            except:
                pass
        print >> sys.stderr, '\r %s / %s' % (i+1,len(frames)),

    print >> sys.stderr, 'DONE'
    return pdmask,pdols,prevactols,newactols
    


def calculate_cumulative_activity(analysis_dir,zcut,suppress_ground=40,shape=(480,720),be=None,write_files=True,force_all=False,xybounds=None,use_ground=None,preshrink=2):
    '''given a directory containing .actmat and .ground files, iteratively calculates new subterranean activity'''
    actfiles = sorted(glob(os.path.join(analysis_dir,'*.actmat')))
    prevact = {}
    prevtermini = {}
    prevprogress = {}
    
    if be is None: be = tuple([n/2 for n in shape])
    
    if use_ground is not None:
        groundmask = mask_from_vector(use_ground+suppress_ground,shape)
    for f in actfiles:
        # set up activity filenames
        newactout = os.path.splitext(f)[0]+'.newactout'
        newactterm = os.path.splitext(f)[0]+'.newactterm'
        preactout = os.path.splitext(f)[0]+'.preactout'
        newactprop = os.path.splitext(f)[0]+'.newactprop'
        reqfiles = [newactout,newactterm,preactout,newactprop]
        
        # skip this iteration (after updating previous activity dicts) if all four outputs are present
        if all([os.path.exists(outfilename) for outfilename in reqfiles]) and not force_all:
            #print >> sys.stderr, 'all output for %s present; skipping' % f
            newout = eval(open(newactout).read())
            newterm = eval(open(newactterm).read())
            pa_out = eval(open(preactout).read()) #not needed
            prop = eval(open(newactprop).read())
            
            prevtermini[f] = newterm
            if len(prevact) == 0:
                prevact['prior'] = pa_out
            else:
                prevact[f] = newout
            prevprogress[f] = prop['progress']
            
            continue
        #else:
            #print >> sys.stderr, 'checking for files',' '.join(reqfiles)
            #for outfilename in reqfiles:
                #if not os.path.exists(outfilename):
                    #print >> sys.stderr, outfilename,'not found'
        #otherwise...
        try:
            actmat = numpy.fromfile(f).reshape(shape)
        except:
            raise ValueError, 'died on %s, shape %s' % (f,shape)
        try: #to find a groundmask
            if use_ground is None:
                groundfile = os.path.splitext(f)[0]+'.ground'
                ground = numpy.fromfile(groundfile,sep='\n')
                groundmask = mask_from_vector(ground+suppress_ground,shape)
            actmat = Util.subtract_mask(actmat,groundmask,0)
        except OSError:
            print >>sys.stderr,'No ground file %s found - proceeding unmasked' % groundfile
        
        #old:
        #pa_flat = flatten_points_lists([v for k,v in sorted(prevact.items()) if len(v) > 0])
        #pt_flat = flatten_points_lists([v for k,v in sorted(prevtermini.items()) if any(v)])
        
        #new -- note, overwrites pt_flat with the historical record of terminus orders,
        # rather than the order of termini from the trace of merged polys.
        # this serves to place the most recent terminus at pt_flat[-1] for subsequent analysis
        if len(reduce(lambda x,y: x+y, prevact.values()+[[]])) == 0:
            pa_flat = [[]]
            prevmask = numpy.zeros(actmat.shape,dtype=bool)
            newact = actmat>zcut
            pa_points = []
        else:
            pa_flat, pt_flat, prevmask = merge_polygons(reduce(lambda x,y: x+y, prevact.values()),shape=shape,start_trace=be)
            newact = Util.subtract_mask(actmat>zcut,prevmask)
            pa_points = flatten_points_lists(pa_flat)
        pt_flat = flatten_points_lists([v for k,v in sorted(prevtermini.items()) if any(v)])
        
        #print >>sys.stderr,'flats:',pa_flat,pt_flat
        
        # HAX 20100320 0 the 5 first and last lines and columns
        newact[:5,:] = 0
        newact[:,:5] = 0
        newact[-5:,:] = 0
        newact[:,-5:] = 0
        
        if pt_flat:
            newout,newterm = chain_outlines_from_mask(newact,pt_flat[-1],preshrink=preshrink)
        else:
            newout,newterm = chain_outlines_from_mask(newact,be,preshrink=preshrink)
        
        if any(newout):
            if pa_points:
                farthest_new = apogee(newout,pa_points)[0]
                #print >> sys.stderr,'farthest new; pa_flat:', farthest_new,pa_flat
                nearest_old,progress = closest_point(pa_points,farthest_new,include_dist=True)
            elif newterm:
                nearest_old = newterm[0][0]
                farthest_new = newterm[-1][-1]
                progress = hypotenuse(nearest_old,farthest_new)
            prevtermini[f] = newterm
            prevact[f] = newout
            prevprogress[f] = progress
            print >> sys.stderr, f
            print >> sys.stderr, 'nearest_old: %s farthest_new: %s progress: %s' % (nearest_old,farthest_new,progress)
        else:
            farthest_new,nearest_old,progress = None,None,None
            
        #UNCOMMENT FOR PLAY-BY-PLAY
        #print >> sys.stderr, f
        #print >> sys.stderr, 'nearest_old: %s farthest_new: %s progress: %s' % (nearest_old,farthest_new,progress)
        
        if write_files:
            #UNCOMMENT FOR OUTFILE NAMES
            #print >> sys.stderr, ('**file output invoked.\nSummary outlines of activity prior to this segment in %s' 
            #	'\nOutlines of activity this segment in %s\nProperties of new activity: %s') \
            #	% (preactout, newactout, newactprop)
            
            #if prevmask.any():
            #	pa_out,pt_null = chain_outlines_from_mask(prevmask,be,preshrink=1,grow_by=2)
            #else:
            #	pa_out = []
            
            prop = {}
            prop['nearest_old'] = nearest_old
            prop['farthest_new'] = farthest_new
            prop['progress'] = progress
            prop['area'] = len(flatten_points_lists(points_from_mask(newact)))
            
            open(newactout,'w').write(newout.__repr__())
            open(newactterm,'w').write(newterm.__repr__())
            open(preactout,'w').write(pa_flat.__repr__())
            open(newactprop,'w').write(prop.__repr__())
            
    return prevact,prevtermini,prevprogress

def calc_activity_segment_bounds(adir,pad_segs=0):
    '''returns list of 2-tuples of seconds (start,end) encompassing activity
    (i.e. non-empty .newactout) in <adir>
    padded by a number of segments <pad_segs>
    
    INCOMPLETE?
    '''
    
    fps,dur = re.search('(\d+)fps.+(\d+)sec',adir).groups()
    
    actoutf = sorted(glob(adir+'/*.newactout'))
    actout = [eval(open(f).read()) for f in actoutf]
    actsegs = numpy.array([len(a) for a in actout])>0
    if pad_segs:
        newas = numpy.zeros(actsegs.shape,dtype=bool)
        for i,v in enumerate(actsegs):
            if v:
                newas[i-pad_segs:i+pad_segs+1] = True
        actsegs = newas
			
    actbounds = Util.get_consecutive_value_boundaries(numpy.arange(len(actout))[actsegs])
    
    actboundsecs = []
    
    for i1,i2 in actbounds: #FINISH!
        start = actoutf[i1]
    
    return actbounds

def mask_from_bounds_list(boundslist,shape):
    '''given a list of xybounds (top left, bottom right, eg [[(x1,y1),(x2,y2)], ... ]), generates a mask True for all encompassed area
    '''
    
    mask = numpy.zeros(shape,dtype=bool)
    for (x1,y1),(x2,y2) in boundslist:
        mask[y1:y2,x1:x2] = True
        
    return mask

def mask_from_vector(vector,shape):
    '''returns a mask that is false "below" (i.e. higher y values) and true "above" (lower y values)
    the value in each item of the vector (x values from vector indices)'''
    
    mask = numpy.zeros(shape[::-1],dtype=bool)
    for i,k in enumerate(vector):
        mask[i,:k] = True
    return mask.transpose()

def mask_band_around_vector(vector,shape,bandheight,bandmasked=False):
    '''returns a mask around a vector
    default returns true outside of band (bandmasked=False), set bandmasked=True to mask inside the band instead
    '''
    halfheight = bandheight/2
    if bandmasked:
        mask = numpy.zeros(shape[::-1],dtype=bool)
        for i,k in enumerate(vector):
            if k-halfheight < 0:
                top = 0
            else:
                top = k-halfheight
            mask[i,top:k+halfheight] = True
    else:
        mask = numpy.ones(shape[::-1],dtype=bool)
        for i,k in enumerate(vector):
            if k-halfheight < 0:
                top = 0
            else:
                top = k-halfheight
            mask[i,top:k+halfheight] = False
    return mask.transpose()

def grow_mask(mask,growby):
	
    values = mask.copy()
    
    for i,r in enumerate(mask):
        for j,k in enumerate(mask[i]):
            if k:
                values[i-growby:i+growby+1,j-growby:j+growby+1] = True
    
    return values

def shrink_mask(mask,shrinkby):
    
    values = numpy.zeros(mask.shape,dtype=bool)
    if shrinkby < 1:
        return mask
    
    for i,r in enumerate(mask):
        for j,k in enumerate(mask[i]):
            if k:
                if not is_edge(mask,(j,i)):
                    values[i,j] = True
    
    if shrinkby-1:
        #print 'shrink %s' % shrinkby
        return shrink_mask(values,shrinkby-1)
    else:
        return values

def merge_polygons(polys,shape=(480,720),start_trace=(360,240),growby=0,preshrink=0):
    '''SHOULD REPLACE WITH PROGRESSIVE UNION FROM SHAPELY
    merges polygons by first generating binary masks of included areas and subsequently tracing the sum area
    
    polys should be a list of lists, which contain tuples 
    consider reduce(lambda x,y: x+y, deep_polys) where deep_polys is a list of lists of lists of tuples, 
    e.g. where elements in the start list are each lists of polygons (which is a list of tuple coords)
    returns:
    the list-of-lists-of-tuples describing topologically distinct polygons
    the termini (nearest points) for each polygon
    the boolean mask of those polys'''
    
    pa_mask = mask_from_outline(polys[0],shape)
    for p in polys[1:]:
        pa_mask += mask_from_outline(p,shape)
    
    sum_poly, sum_term = chain_outlines_from_mask(pa_mask,start_trace,grow_by=growby+preshrink,preshrink=preshrink,border=0)
    return sum_poly, sum_term, pa_mask

def flatten_points_lists(coords_lists):
    try:
        if isinstance(coords_lists[0],tuple):
            return coords_lists
        elif isinstance(coords_lists[0],list):
            return flatten_points_lists(reduce(lambda x,y: x+y,coords_lists))
    except IndexError:
        return coords_lists

def hypotenuse(p1,p2):
    if not (len(p1) == 2 and len(p2) == 2):
        print >> sys.stderr, '%s and %s must be (x,y) points' % (p1,p2)
        raise TypeError, 'invalid arguments'
    import math
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return math.sqrt(x**2 + y**2)

def line_fx_from_pts(p1,p2):
    '''returns m and b for a line through two points supplied'''
    run,rise = numpy.array(p1)-numpy.array(p2)
    if run == 0:
        run = 10e-3
    m = rise/run
    b = p1[1]-(m*p1[0])
    return m,b

def distance_from_point(coords,point):
    '''given a list of coord 2-tuples and a 2-tuple point, returns a list of coords (None for None values in coords)'''
    
    dists = []
    for c in coords:
        if c is None:
            dists.append(c)
        else:
            dists.append(hypotenuse(c,point))
    return dists

def distance_from_poly(poly,point):
    '''given a list of coord 2-tuples comprising a single polygon,
    or a list of lists of coord 2-tuples comprising many polygons,
    finds the distance to the nearest polygon POINT (not edge)
    returns 0 if the point is in any of the polygons'''
    
    if isinstance(poly[0],tuple):
        xs,ys = Util.dezip(poly+[point])
        actmat = mask_from_outline(poly,(max(ys)+10,max(xs)+10))
        if actmat[point[1],point[0]] == True:
            return 0
        else:
            return closest_point(poly,point,include_dist=True)[1]
    else:
        return min([distance_from_poly(p,point) for p in poly])

def closest_point(coords,point,include_dist=False):
    '''returns the closest (x,y) point from the supplied list'''
    flatcoords = flatten_points_lists(coords)
    idx = numpy.array(distance_from_point(flatcoords,point)).argmin()
    if include_dist:
        return flatcoords[idx],hypotenuse(flatcoords[idx],point)
    else:
        return flatcoords[idx]

def farthest_point(coords,point,include_dist=False):
    '''returns the farthest (x,y) point from the supplied list'''
    flatcoords = flatten_points_lists(coords)
    idx = numpy.array(distance_from_point(coords,point)).argmax()
    if include_dist:
        return flatcoords[idx],hypotenuse(flatcoords[idx],point)
    else:
        return flatcoords[idx]

def point_distance_matrix(coords1,coords2):
    mat = numpy.zeros((len(coords1),len(coords2)))
    for i,c1 in enumerate(coords1):
        for j,c2 in enumerate(coords2):
            mat[i,j] = hypotenuse(c1,c2)
    return mat

def apogee(coords1,coords2,include_dist=False):
    #if isinstance(coords1[0],list):
    #	coords1 = reduce(lambda x,y: x+y, coords1)
    #if isinstance(coords2[0],list):
    #	coords2 = reduce(lambda x,y: x+y, coords2)
    coords1 = flatten_points_lists(coords1)
    coords2 = flatten_points_lists(coords2)
    
    dmat = point_distance_matrix(coords1,coords2)
    c2,c1 = xy_from_idx(dmat.argmax(),dmat.shape) #c2/c1 reversal compensates for x,y <-> i,j mapping reversal
    return coords1[c1],coords2[c2]

def perigee(coords1,coords2,include_dist=False):
    if isinstance(coords1[0],list):
        coords1 = reduce(lambda x,y: x+y, coords1)
    if isinstance(coords2[0],list):
        coords2 = reduce(lambda x,y: x+y, coords2)
    
    dmat = point_distance_matrix(coords1,coords2)
    c2,c1 = xy_from_idx(dmat.argmin(),dmat.shape) #c2/c1 reversal compensates for x,y <-> i,j mapping reversal
    return coords1[c1],coords2[c2]

def points_from_mask(mask):
    '''returns a list of (x,y) coords lying in "true" mask cells'''
    flatmask = mask.flatten()
    flat_idxs = numpy.arange(0,len(flatmask))[flatmask]
    return [xy_from_idx(i,mask.shape) for i in flat_idxs]

def get_adjacent_values(mask,point,skip=[],xybounds=None):
    '''returns a list of values (clockwise from top) adjacent to the specified point'''
    xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
    maxy,maxx = mask.shape
    return [mask[point[1]+y,point[0]+x] for x,y in xys if (x,y) not in skip and 0<point[1]+y<maxy and 0<point[0]+x<maxx]

def get_adjacent_points(point):
    xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
    return [(point[0]+x,point[1]+y) for x,y in xys]

def get_next_edge(mask,point,prev_edges,xybounds=None):
    xys = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
    if not isinstance(prev_edges,list):
        prev_edges = [prev_edges]
    #print >> sys.stderr,'next point from',point,'avoiding', prev_edges
    skip = [(prev_edge[0]-point[0],prev_edge[1]-point[1]) for prev_edge in prev_edges]
    for i,val in enumerate(get_adjacent_values(mask,point)):
        if val and not xys[i] in skip:
            candidate = (point[0] + xys[i][0], point[1] + xys[i][1])
            #print >> sys.stderr,'%s not in %s, edge?' % (candidate,skip)
            if mask[candidate[1],candidate[0]] and is_edge(mask,candidate):
                #print >> sys.stderr,'yes; keeping'
                return candidate

def outline_from_mask(mask,origin=(0,0),grow_by=1,preshrink=0,xybounds=None):
    '''given a mask and an origin (point to start closest to) returns an ordered list of (x,y) that trace the mask'''
    if preshrink:
        mask = shrink_mask(mask,preshrink)
        if grow_by:
            grow_by += preshrink
    if grow_by:
        mask = grow_mask(mask,grow_by)
    mask[:2,:] = False
    mask[-2:,:] = False
    mask[:,:2] = False
    mask[:,-2:] = False
    
    if not mask.any(): return []
    
    start = closest_edge(mask,origin)
    #print >> sys.stderr, 'start at %s' % (start,)
    outline = [start]
    next = get_next_edge(mask,start,start)
    while next and next != outline[0]:
        #print >>sys.stderr, 'next point: %s' % (next,)
        outline.append(next)
        try:
            invalid = outline[2:-1]+get_adjacent_points(outline[-3])#+get_adjacent_points(outline[-2])
        except IndexError:
            invalid = outline[-4:-1]
        next = get_next_edge(mask,outline[-1],invalid)
    return outline

def order_points(pts,mask):
    '''given a polygon of unordered points and the underlying mask e.g. from chain_outlines_from_mask below
    returns a list of ordered points (suitable for plotting)'''
    p0 = pts.pop(0)
    outline = [p0]
    left = set(pts)
    while left and min([hypotenuse(p0,p) for p in list(left)]) == 1:
        p0 = sorted(list(left),key=lambda x: (hypotenuse(x,p0),mask[x[1]-5:x[1]+5,x[0]-5:x[0]+5].sum()))[0]
        outline.append(p0)
        left -= set([p0])
    return outline


def chain_outlines_from_mask_shapely(mask,grow_by=0,preshrink=1,xybounds=None,border=5,interp_dist=0.01,\
                                     preserve_topology=True,debug=False):
    

    if debug:
        now = time.time()
        start = now

    
    if border > 0:
        ydim,xdim = mask.shape
        newmask = numpy.zeros((ydim+2*border,xdim+2*border),dtype='bool')
        newmask[border:-border,border:-border] = mask
    else:
        newmask = mask.copy()

    if debug:
        print time.time()-now,'set border'	
        now = time.time()
    
    if preshrink:
        newmask = shrink_mask(newmask,preshrink)
        newmask = grow_mask(newmask,preshrink)
    if grow_by:
        newmask = grow_mask(newmask,grow_by)

    if debug:
        print time.time()-now,'shrink/grow'	
        now = time.time()
    
    #no need to proceed if no blobs to trace!
    if not newmask.any():
        return []

    pts = points_from_mask(newmask)
    polys = MultiPoint(pts).buffer(1).buffer(-1).simplify(0.01,preserve_topology=preserve_topology)

    if debug:
        print time.time()-now,'trace blobs'	
        now = time.time()
    
    ols = []
    try:
        if isinstance(polys,Polygon):
            polys = [polys]
        for poly in polys:
            ols.append([])
            for x,y in numpy.array(poly.exterior):
                p = (int(round(x)),int(round(y)))
                if len(ols[-1]) == 0 or ols[-1][-1] != p:
                    ols[-1].append(p)
    except TypeError:
        print >> sys.stderr, 'type of polys is %s looks like:\n\n%s' % (type(polys),polys)
        raise

    if debug:
        print time.time()-now,'filter traces'	
        now = time.time()

    resetouts = []
    for p in ols:
        resetp = []
        for x,y in p:
            resetp.append((x-border,y-border))
        resetouts.append(resetp)

    if debug:
        print time.time()-now,'reset outlines'	
        now = time.time()
        print 'total %s' % (now-start)
        
    return resetouts

def chain_outlines_from_mask(mask,origin=(0,0),grow_by=0,preshrink=1,xybounds=None,border=5,\
			     return_termini=True,order_points=True,sort_outlines=True,debug=False):
    '''given a mask returns a list of polygons (lists of (x,y) point tuples)
    which outline each discrete solid object in the mask
    and a list of lists of termini which define progress in each interval
    
    xybounds not currently implemented'''
    if debug:
        now = time.time()
        start = now
    
    if border > 0:
        ydim,xdim = mask.shape
        newmask = numpy.zeros((ydim+2*border,xdim+2*border),dtype='bool')
        newmask[border:-border,border:-border] = mask
    else:
        newmask = mask.copy()
    
    if debug:
        print time.time()-now,'set border'	
        now = time.time()
    
    if preshrink:
        newmask = shrink_mask(newmask,preshrink)
        newmask = grow_mask(newmask,preshrink)
    if grow_by:
        newmask = grow_mask(newmask,grow_by)
    
    if debug:
        print time.time()-now,'shrink/grow'	
        now = time.time()

    #no need to proceed if no blobs to trace!
    if not newmask.any():
        if return_termini:
            return [],[]
        else:
            return []
	
    olps = [p for p in points_from_mask(newmask) if is_edge(newmask,p)]
    olps.sort()
    
    if debug:
        print time.time()-now,'outline pts'	
        now = time.time()
    
    adj_pts = [[olps[0]]]
    for p in olps[1:]:
        found = False
        drops = []
        for i,pli in enumerate(adj_pts):
            if any([hypotenuse(p,plip) <= 1 for plip in pli]):
                if found:
                    adj_pts[found].extend(pli)
                    drops.append(i)
                else:
                    pli.append(p)
                    found = i
        dropped = [adj_pts.pop(popi) for popi in drops]
        if found == False:
            adj_pts.append([p])
    
    adj_pts.sort(key=len,reverse=True)
    
    if debug:
        print time.time()-now,'into bloblists'
        now = time.time()
    
    nonol = [adj_pts[0]]
    for poly in adj_pts:
        if not any([mask_from_outline(inpol,newmask.shape)[mask_from_outline(poly,newmask.shape)].any() for inpol in nonol]):
            nonol.append(poly)
    
    #nonol[0][:0] = olps[:1]
    
    if debug:
        print time.time()-now,'get nonoverlapping'	
        now = time.time()
    
    if order_points:
        outlines = []
        for pts in nonol:
            p0 = pts.pop(0)
            outline = [p0]
            left = set(pts)
            while left and min([hypotenuse(p0,p) for p in list(left)]) == 1:
                p0 = sorted(list(left),key=lambda x: (hypotenuse(x,p0),newmask[x[1]-5:x[1]+5,x[0]-5:x[0]+5].sum()))[0]
                outline.append(p0)
                left -= set([p0])
            outlines.append(outline)
    else:
        outlines = nonol
    
    if debug:
        print time.time()-now,'order points'	
        now = time.time()
    
    if sort_outlines:
        outlines.sort(key = lambda x: hypotenuse(origin,closest_point(x,origin)) )
        sortouts = [outlines.pop(0)]
        while outlines:
            outlines.sort(key = lambda x: hypotenuse( *perigee(x,sortouts[-1]) ))
            sortouts.append(outlines.pop(0))
            
        outlines = sortouts
    
    if debug:
        print time.time()-now,'sort outlines'	
        now = time.time()
    
    if return_termini:
        spanshapes = [[origin]]+outlines
        termini = []
        for i in range(1,len(spanshapes)-1):
            termini.append([perigee(spanshapes[i-1],spanshapes[i])[1],perigee(spanshapes[i],spanshapes[i+1])[0]])
        last_entry = perigee(spanshapes[-2],spanshapes[-1])[1]
        termini.append([last_entry,farthest_point(spanshapes[-1],last_entry)])
    
    if debug:
        print time.time()-now,'define termini'
        now = time.time()
    
    resetouts = []
    resetterms = []
    for p in outlines:
        resetp = []
        for x,y in p:
            resetp.append((x-border,y-border))
        resetouts.append(resetp)
    if return_termini:
        for p in termini:
            resetp = []
            for x,y in p:
                resetp.append((x-border,y-border))
            resetterms.append(resetp)
    
    if debug:
        print time.time()-now,'fix border'	
        now = time.time()
        print 'total %s' % (now-start)	
        
    if return_termini:
        return resetouts,resetterms
    else:
        return resetouts

def outline_overlap(ol1,ol2,SHAPE):
    m1 = mask_from_outline(ol1,SHAPE)
    m2 = mask_from_outline(ol2,SHAPE)
    return len(filter(None,m1[m2]))

def find_objs(ols,SHAPE,num_obj_min=1,num_obj_max=numpy.inf,conflicts='break'):
    '''given a list of lists of point-lists representing polygons outlining segmented blobs \
    (e.g. from chain_oulines_from_mask run on consecutive frames of video)

    returns a list of lists, each inner list represents an "object arc"; a contiguous overlapping series of polygons
    inner list elements are 2-tuples of (i,j) where i is index of outer ols list (e.g. frame number), j is index of polygon in list ols[i]

    num_obj_[min|max] sets the number of blobs (inclusive) that must be present in a frame (e.g. length of list ols[i])
    to consider inclusion in resulting object arcs
    
    conflicts:
    A conflict is defined as either
    the last blob of a single object arc overlapping two or more blobs in the current frame (a split)
    -OR-
    a single blob in the current frame overlaps the last blobs of two or more object arcs (a join)

    conflict resolution modes:
    "break":
        new object arc[s] are created, such that a split ends one and starts two or more new,
        and a join ends two or more and starts one new
    "choose":
        resolves branchpoints by selecting the longest of the conflicting arcs. If tied, the largest object wins
        non-chosen conflicting arcs are considered separate objects
    "merge":
        based on a size distribution of pre-split and/or post-join single blobs, attempts to merge conflicting object arcs into
        size-consistent single blobs in each conflicted frame.
        Falls back to "choose" if conflicting object arcs are at any point separated by more than 2 standard deviations
        of pre-split and/or post-join object size derived diameter distribution
    '''

    def get_split(ti,l,tli):
        return [set(l).intersection(set(tl)) for tj,tl in enumerate(tli) if ti != tj and len(set(l).intersection(set(tl))) > 0]
    
    if conflicts not in ['break','choose','merge']:
        raise ValueError, 'conflicts (conflict resolution mode) must be one of "break" "choose" "merge"' 
    
    objs = []
    joins = {}
    splits = defaultdict(list)
    for i,ol in enumerate(ols):
        if len(ol) < num_obj_min or len(ol) > num_obj_max:
            continue
        idxs = []
        for j,p in enumerate(ol):
            idxs.append([])
            for obj_idx,prev_ols in enumerate(objs):
                last_fr,last_olid = prev_ols[-1]
                last_p = ols[last_fr][last_olid]
                #print >> sys.stderr, i,j,last_fr,last_olid,idxs,objs,len(last_p)
                if last_fr == (i-1) and outline_overlap(last_p, p, SHAPE) > 0:
                    idxs[-1].append(obj_idx)

        for j,obj_ovl in enumerate(idxs):
            if len(obj_ovl) == 0:
                objs.append([(i,j)])
            elif len(obj_ovl) == 1:
                if any(get_split(j,idxs[j],idxs)):
                    for prev_obj in reduce(set.union,get_split(j,idxs[j],idxs)):
                        splits[prev_obj].append(len(objs))
                    objs.append([(i,j)])
                else:
                    objs[obj_ovl[0]].append((i,j))
            else:
                if any(get_split(j,idxs[j],idxs)):
                    for prev_obj in reduce(set.union,get_split(j,idxs[j],idxs)):
                        splits[prev_obj].append(len(objs))
                joins[len(objs)] = obj_ovl
                objs.append([(i,j)])

    return objs,splits,joins

def resolve_objs_join(fork_obj_arc_ids,idxs_replace,ols,ols_offset,start_frame, next_offset, SHAPE,objs,splits,objs_sizes,objs_fols,size_h,size_bins,fol_h,fol_bins):
    '''OBJS IS DICT (not list)
    obj_arc_ids is a list of object IDs (keys in objs) that need
    resolution by joining (locally lowered threshold will be implemented here)

    #for now, attach best fork arc to pre-split object arc, remove split
    # return pre-split obj arc id
    # OR if no split, then return best fork arc
 
    '''
    #best_arc_id = sorted(fork_obj_arc_ids, key = lambda x: score_object_arc_from_values(objs_sizes[x],objs_fols[x],size_h,size_bins,fol_h,fol_bins))[-1]
    #print >> sys.stderr, 'best arc',best_arc_id

    ps_obj_arc_id_candidates = [oid for oid,slist in splits.items() if set(slist) >= set(fork_obj_arc_ids)]

    if len(ps_obj_arc_id_candidates) == 1:
        ps_obj_arc_id = ps_obj_arc_id_candidates[0]
        #print >> sys.stderr, 'forks',fork_obj_arc_ids,'match one root',ps_obj_arc_id

        best_arc_id = merge_obj_arc_data(ps_obj_arc_id, fork_obj_arc_ids, idxs_replace, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)

        splits[ps_obj_arc_id] = [oid for oid in splits[ps_obj_arc_id] if oid not in fork_obj_arc_ids]
        if len(splits[ps_obj_arc_id]) == 0:
            #print >> sys.stderr, 'no remaining forks for split',ps_obj_arc_id
            splits.pop(ps_obj_arc_id)

        return ps_obj_arc_id
    else:
        #give back highest scoring choice, if no corresponding fork
        return sorted(fork_obj_arc_ids, key = lambda x: score_object_arc_from_values(objs_sizes[x],objs_fols[x],size_h,size_bins,fol_h,fol_bins))[-1] 

def ol_ids_to_relative(ol_ids,ols_offset):
    '''
    takes an object arc list (list of 2-tuples representing absolute frame and blob number in that frame)
    returns list with frame numbers converted to relative coordinates of an outline set of specified <offset>
    '''
    rel_li = []
    for fr,oln in ol_ids:
        rel_li.append((fr-ols_offset,oln))
    return rel_li

def merge_obj_arc_data(parent_obj_id, candidate_child_obj_ids, idxs_replace, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins):
    '''parent is a single object arc id to merge onto
    child ids are a list of potential additions; the highest-scoring will be chosen
    '''
    best_arc_id = sorted(candidate_child_obj_ids, key = lambda x: score_object_arc_from_values(objs_sizes[x],objs_fols[x],size_h,size_bins,fol_h,fol_bins))[-1]

    #handle missing overlap between last of pre-split and first of best_arc
    #print >> sys.stderr, 'parent_obj_id: %s best_arc_id: %s\n\tsplits dict: %s parent len: %s' % (parent_obj_id,best_arc_id,splits,len(objs[parent_obj_id]))

    missing_ol = fol_from_obj_arc(ol_ids_to_relative([objs[parent_obj_id][-1],objs[best_arc_id][0]],ols_offset),ols,SHAPE)[0]

    objs[parent_obj_id].extend(objs.pop(best_arc_id))
    #remove child arc from list of forks
    splits[parent_obj_id] = filter(lambda x: x!=best_arc_id, splits[parent_obj_id])
    #if child is itself a split parent, transfer child's downstream forks to parent
    if best_arc_id in splits:
        #print >> sys.stderr, 'splits dict: %s\n\tchild %s to be merged onto %s' % (splits,best_arc_id,parent_obj_id)
        splits[parent_obj_id] = splits.pop(best_arc_id)
        #print >> sys.stderr, '\t new value: %s' % splits[parent_obj_id]
        #print >> sys.stderr, '\tremaining splits keys: %s' % splits.keys()
    objs_sizes[parent_obj_id].extend(objs_sizes.pop(best_arc_id))
    objs_fols[parent_obj_id].extend([missing_ol] + objs_fols.pop(best_arc_id))

    idxs_replace[best_arc_id] = parent_obj_id
    #print >> sys.stderr, '\tidxs_replace: %s' % idxs_replace

    return best_arc_id

def get_retired_obj_ids(current_frame_idx, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins):
    retire_obj_ids = []

    for obj_id,obj_arc in sorted(objs.items(), key = lambda (k,v): score_object_arc_from_values(objs_sizes[k],objs_fols[k],size_h,size_bins,fol_h,fol_bins)):
        if obj_arc[-1][0] < (ols_offset + current_frame_idx) and (not obj_id in splits.keys() or len(splits[obj_id]) == 0):
            #most recent element is not current; this is a finished arc
            resolved_splits = []
            for ps_obj_arc_id,fork_obj_arc_ids in splits.items():
                if obj_id in fork_obj_arc_ids:
                    #print >> sys.stderr, 'object arc %s is out of date; most recent addition is %s; this is iteration %s' % (obj_id,obj_arc[-1],current_frame_idx+ ols_offset)
                    drop_idx = fork_obj_arc_ids.index(obj_id)
                    fork_obj_arc_ids.pop(drop_idx)
                    if obj_id in splits.keys():
                        splits.pop(obj_id)
            retire_obj_ids.append(obj_id)

    return retire_obj_ids

def cleanup_splits(ols, ols_offset, idxs_replace, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins):
    resolved_splits = []
    for k,v in splits.items():
        if len(v) == 0:
            splits.pop(k)
            
    for ps_obj_arc_id,fork_obj_arc_ids in sorted(splits.items(), key=lambda (k,v): v[-1][0], reverse=True):
        #print >> sys.stderr, 'evaluate',ps_obj_arc_id,'splits:',splits[ps_obj_arc_id],'object len:',len(objs[ps_obj_arc_id]),'splits:',splits.items()
        if not ps_obj_arc_id in splits:
            #print >> sys.stderr, 'the current key %s has been removed from splits dict during iteration; SKIP' % ps_obj_arc_id
            continue
        #print >> sys.stderr, ps_obj_arc_id,'reprocess forked object',fork_obj_arc_ids,'with',objs
        fork_obj_arc_ids = [oid for oid in fork_obj_arc_ids if oid in objs.keys()]
        #print >> sys.stderr, 'as',fork_obj_arc_ids
        if len(fork_obj_arc_ids) == 1:
            kept_fork = merge_obj_arc_data(ps_obj_arc_id, fork_obj_arc_ids, idxs_replace, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)
            resolved_splits.append(ps_obj_arc_id)
        if len(fork_obj_arc_ids) == 0:
            resolved_splits.append(ps_obj_arc_id)
    #old drop:
    #for splt in resolved_splits:
    #    splits.pop(splt)
    #new drop:
    for split_key in splits.keys():
        if len(splits[split_key]) == 0 or split_key in resolved_splits:
            splits.pop(split_key)

def resolve_idxs_replace_obj(obj_ovl,idxs_replace):
    #print >> sys.stderr, 'obj_ovl intial      : %s' % (obj_ovl)
    obj_ovl_rep = [idxs_replace.get(oo,oo) for oo in obj_ovl]
    #print >> sys.stderr, 'obj_ovl post-replace: %s' % (obj_ovl_rep)
    return obj_ovl_rep

def resolve_idxs_replace_idxs(idxs,idxs_replace):
    #print >> sys.stderr, 'idxs intial      : %s' % (idxs)
    idxs_rep = [resolve_idxs_replace_obj(obj_ovl,idxs_replace) for obj_ovl in idxs]
    #print >> sys.stderr, 'idxs post-replace: %s' % (idxs_rep)
    return idxs_rep

def find_objs_progressive(ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins, num_obj_min=1 ,num_obj_max=numpy.inf):
    '''OBJS IS DICT (not list)
    extends an existing object arc dictionary given a list of outlines and the following:

    ols : list of outline polygon lists 
        (outer list is frame-by-frame; inner lists are blob outlines for that frame)
    ols_offset : absolute frame number of 0-th frame represented in <ols>
    start_frame : absolute frame number to start interation at in <ols>
                    (iteration will start at ols[start_frame-ols_offset])
    next_offset : absolute frame number of the first (lowest index) frame from which to allow a split to persist.  Set to the anticipated <ols_offset> of the NEXT invocation of find_obs_progressive() to ensure splits are resolved while the underlying frame/outline data are available in memory
    SHAPE : LxW pixel dimensions of video frames
    objs : object arc dictionary to present
    splits : dictionary of "split" object arcs (frame n-1 blob overlaps two frame n blobs; dict is {(frame n-1, blob_id) : [(frame n, blob_1_id),(frame n, blob_2_id)] })
    objs_sizes : dictionary of blob sizes lists (in pixels) by obj_arc_ids
    objs_fols : dictionary of blob fraction_overlap lists (in pixels) by obj_arc_ids
    size_h, size_bins, fol_h, fol_bins : scoring distribution parameters from get_object_arc_param_dists_from_values_dict() or get_object_arc_param_dists()

    return retired_objs, retired_objs_sizes, retired_objs_fols
    retired_objs, sizes list and fraction_overlap list of object arcs that were "retired" this invocation

    '''

    def get_split(ti,l,tli):
        return [set(l).intersection(set(tl)) for tj,tl in enumerate(tli) if ti != tj and len(set(l).intersection(set(tl))) > 0]

    retired_objs = {}
    retired_objs_sizes = {}
    retired_objs_fols = {}


    start_index = start_frame - ols_offset #relative index of absolute start in current ols
    for i,ol in enumerate(ols[start_index:],start_index):
        #print >> sys.stderr, '\nindex %s absolute frame %s start' % (i,i+ols_offset)
        #print >> sys.stderr, '\tobjs lengths,last: %s' % [(k,len(v),v[-1]) for k,v in objs.items()]
        #print >> sys.stderr, '\tsplits dict: %s' % splits
        
        idxs_replace = {}
        if len(ol) < num_obj_min or len(ol) > num_obj_max:
            continue
        idxs = []
        for j,p in enumerate(ol): #find all overlaps with most recent frame incorporated in objects
            idxs.append([])
            for obj_id,prev_ols in objs.items():
                last_fr,last_olid = ol_ids_to_relative(prev_ols,ols_offset)[-1]
                last_p = ols[last_fr][last_olid]
                #print >> sys.stderr, last_fr, i, outline_overlap(last_p, p, SHAPE)
                if last_fr == (i-1) and outline_overlap(last_p, p, SHAPE) > 0:
                    #if this object was in the previous frame, 
                    # and it overlaps the current outline,
                    # record under the appropriate outline index in <idxs>
                    idxs[-1].append(obj_id)

        #print >> sys.stderr, 'matches',idxs
        for j,obj_ovl in enumerate(idxs): #handle overlaps
            
            #print >> sys.stderr, 'obj arc IDs',objs.keys(), 'j=',j
            if len(obj_ovl) == 0:
                # this is a new object (no prior observation)
                objs[(i+ols_offset,j)] = [(i+ols_offset,j)]
                objs_sizes[(i+ols_offset,j)] = [size_of_polygon(ols[i][j])]
                objs_fols[(i+ols_offset,j)] = []
            else:
                if len(obj_ovl) != 1:
                    #resolve as join; see resolve_objs_join
                    #print >> sys.stderr,'resolve join',obj_ovl
                    obj_id_to_append = resolve_objs_join( resolve_idxs_replace_obj(obj_ovl,idxs_replace),idxs_replace,ols,ols_offset,start_frame, next_offset, SHAPE,objs,splits,objs_sizes,objs_fols,size_h,size_bins,fol_h,fol_bins)
                else:
                    obj_id_to_append = resolve_idxs_replace_obj(obj_ovl,idxs_replace)[0]
                #handle potential split
                if any(get_split(j,idxs[j],idxs)):
                    objs[(i+ols_offset,j)] = [(i+ols_offset,j)]
                    objs_sizes[(i+ols_offset,j)] = [size_of_polygon(ols[i][j])]
                    objs_fols[(i+ols_offset,j)] = []
                    for prev_obj in reduce(set.union,get_split(j,resolve_idxs_replace_idxs(idxs,idxs_replace)[j],resolve_idxs_replace_idxs(idxs,idxs_replace))):
                        splits[prev_obj].append((i+ols_offset,j))
                else:
                    objs[obj_id_to_append].append((i+ols_offset,j))
                    objs_sizes[obj_id_to_append].append(size_of_polygon(ols[i][j]))
                    objs_fols[obj_id_to_append].append(fol_from_obj_arc(ol_ids_to_relative(objs[obj_id_to_append][-2:],ols_offset),ols,SHAPE)[0])

            #print >> sys.stderr, 'index %s absolute frame %s blob %s handling complete' % (i,i+ols_offset,j)
            #print >> sys.stderr, '\tobjs lengths,last: %s' % [(k,len(v),v[-1]) for k,v in objs.items()]
            #print >> sys.stderr, '\tsplits dict: %s' % splits
            #print >> sys.stderr, '\tidxs_replace dict: %s' % idxs_replace


        #print >> sys.stderr, objs,objs_sizes,objs_fols
        #evaluate current object set and splits to see if any objects can be retired
        retire_obj_ids = get_retired_obj_ids(i, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)

        for obj_id in retire_obj_ids:
            retired_objs[obj_id] = objs.pop(obj_id)
            retired_objs_sizes[obj_id] = objs_sizes.pop(obj_id)
            retired_objs_fols[obj_id] = objs_fols.pop(obj_id)

        #print >> sys.stderr, 'index %s absolute frame %s before cleanup_splits' % (i,i+ols_offset)
        #print >> sys.stderr, '\tobjs lengths,last: %s' % [(k,len(v),v[-1]) for k,v in objs.items()]
        #print >> sys.stderr, '\tsplits dict: %s' % splits


        cleanup_splits(ols, ols_offset, idxs_replace, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)
        #print >> sys.stderr, 'obj arc IDs',objs.keys(),'splits:',splits.__repr__(),'end interation',i

    #no point keeping any split too old to still have primary data loaded on next iteration:
    for obj_id,candidate_child_obj_ids in splits.items():
        if obj_id[0] < next_offset:
            kept_fork = merge_obj_arc_data(obj_id, candidate_child_obj_ids, idxs_replace, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)
            splits.pop(obj_id)

    #one final pass to clear last retirees    
    retire_obj_ids = get_retired_obj_ids(i, ols, ols_offset, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)

    for obj_id in retire_obj_ids:
        retired_objs[obj_id] = objs.pop(obj_id)
        retired_objs_sizes[obj_id] = objs_sizes.pop(obj_id)
        retired_objs_fols[obj_id] = objs_fols.pop(obj_id)
  
    #one last cleanup of the splits list
    cleanup_splits(ols, ols_offset, idxs_replace, start_frame, next_offset, SHAPE, objs, splits, objs_sizes, objs_fols, size_h,size_bins,fol_h,fol_bins)

    return retired_objs, retired_objs_sizes, retired_objs_fols






def object_arc_size_values(ols,SHAPE,n_obj = 1,objs_this_train=None):
    if objs_this_train is None:
        objs_this_train,splits_this_train,joins_this_train = find_objs(ols,SHAPE,n_obj,n_obj)
    size_values = reduce(lambda x,y:x+y,[[size_of_polygon(ols[i][j]) for i,j in o] for o in objs_this_train if len(o) > 1])

    return size_values

def object_arc_fol_values(ols,SHAPE,n_obj = 1,objs_this_train=None):
    if objs_this_train is None:
        objs_this_train,splits_this_train,joins_this_train = find_objs(ols,SHAPE,n_obj,n_obj)
    fol_values = reduce(lambda x,y:x+y,[fol_from_obj_arc(o,ols,SHAPE) for o in objs_this_train if len(o) > 1])

    return fol_values

def get_object_arc_param_dists_from_values_dict(size_train_values_d,fol_train_values_d,size_binw=50,fol_binw=0.05):
    size_values = reduce(lambda x,y:x+y,reduce(lambda x,y:x+y,size_train_values_d.values()))
    fol_values = reduce(lambda x,y:x+y,reduce(lambda x,y:x+y,fol_train_values_d.values()))
    #for offset in size_train_values_d.keys():
    #    if size_train_values_d[offset]:
    #        size_values.extend(reduce(lambda x,y:x+y,size_train_values_d[offset]))
    #        fol_values.extend(reduce(lambda x,y:x+y,fol_train_values_d[offset]))
        
    max_size = max(size_values)+(2*max((size_binw,100)))
    size_h,size_bins = numpy.histogram(size_values,bins = numpy.arange(0,max_size,size_binw))

    fol_h,fol_bins = numpy.histogram(fol_values,bins = numpy.arange(0,1+(2*fol_binw),fol_binw))
    
    return size_h,size_bins,fol_h,fol_bins

def get_object_arc_param_dists(ols,SHAPE,n_obj = 1,size_binw=50,fol_binw=0.05):
    
    objs_this_train,splits_this_train,joins_this_train = find_objs(ols,SHAPE,n_obj,n_obj)

    if len([o for o in objs_this_train if len(o) > 1]) == 0:
        return None,None,None,None

    max_size = max(reduce(lambda x,y:x+y,[[size_of_polygon(p) for p in ol if len(p) > 0] for ol in ols if len(ol) > 0]))+(2*max((size_binw,100)))
    size_values = reduce(lambda x,y:x+y,[[size_of_polygon(ols[i][j]) for i,j in o] for o in objs_this_train if len(o) > 1])
    size_h,size_bins = numpy.histogram(size_values,bins = numpy.arange(0,max_size,size_binw))

    fol_values = reduce(lambda x,y:x+y,[fol_from_obj_arc(o,ols,SHAPE) for o in objs_this_train if len(o) > 1])
    fol_h,fol_bins = numpy.histogram(fol_values,bins = numpy.arange(0,1+(2*fol_binw),fol_binw))

    return size_h,size_bins,fol_h,fol_bins

def get_hbin(val,bins):
    for hbin,binval in enumerate(bins):
        if hbin+1 == len(bins):
            return hbin
        elif binval <= val < bins[hbin+1]:
            return hbin

def fol_from_obj_arc(obj_arc,ols,SHAPE):
    fol = []
    i,j = obj_arc[0]
    prevo = ols[i][j]
    for i,j in obj_arc[1:]:
        fol.append(outline_overlap(prevo,ols[i][j],SHAPE)/float(size_of_polygon(ols[i][j])))
        prevo = ols[i][j]
    return fol

def score_object_arc_size(obj_arc,ols,size_h,size_bins):
    objs_size_scores = [size_h[get_hbin(size_of_polygon(ols[i][j]),size_bins)]+1 for i,j in obj_arc]
    return sum(objs_size_scores)

def score_object_arc_fol(obj_arc,ols,fol_h,fol_bins,SHAPE):
    objs_fol_scores = [fol_h[get_hbin(fol,fol_bins)]+1 for fol in fol_from_obj_arc(obj_arc,ols,SHAPE)]
    return sum(objs_fol_scores)

def score_object_arc(obj_arc,ols,size_h,size_bins,fol_h,fol_bins,SHAPE):
    size_score = score_object_arc_size(obj_arc,ols,size_h,size_bins)
    fol_score = score_object_arc_fol(obj_arc,ols,fol_h,fol_bins,SHAPE)
    return size_score + fol_score

def score_object_arc_from_values(size_values,fol_values,size_h,size_bins,fol_h,fol_bins):
    objs_size_scores = []
    for s in size_values:
        try:
            objs_size_scores.append(size_h[get_hbin(s,size_bins)]+1)
        except IndexError:
            #print >> sys.stderr, 'value %s exceeds maximum size' % (s)
            objs_size_scores.append(min(size_h)+1)
    #list comp replaced by stepwise array construction to handle index errors
    #objs_size_scores = [size_h[get_hbin(s,size_bins)]+1 for s in size_values]
    objs_fol_scores = []
    for f in fol_values:
        try:
            objs_fol_scores.append(fol_h[get_hbin(f,fol_bins)]+1)
        except IndexError:
            print >> sys.stderr, 'value %s exceeds maximum size? hbin: %s' % (f, get_hbin(f,fol_bins))
    #objs_fol_scores = [fol_h[get_hbin(f,fol_bins)]+1 for f in fol_values]
    return sum(objs_size_scores)+sum(objs_fol_scores)

def filter_objs_arcs_by_cutoff(objs_sizes,objs_fols,min_arc_score,size_h, size_bins, fol_h, fol_bins):
    return sorted([k for k in objs_sizes if score_object_arc_from_values(objs_sizes[k],objs_fols[k], size_h, size_bins, fol_h, fol_bins) >= min_arc_score],key = lambda x: score_object_arc_from_values(objs_sizes[x],objs_fols[x], size_h, size_bins, fol_h, fol_bins) )

def arc_segment_in_interval(obj_arc,start,end):
    return [(fr,bl) for fr,bl in obj_arc if start <= fr < end]

def ols_in_interval(start,end,min_arc_score,ols,ols_offset,objs,objs_sizes,objs_fols,size_h, size_bins, fol_h, fol_bins,return_dict=False):                                         
    obj_arcs_above_cutoff = filter_objs_arcs_by_cutoff(objs_sizes,objs_fols,min_arc_score,size_h, size_bins, fol_h, fol_bins)
    if return_dict:
        ols_in_bounds = defaultdict(list)
    else:
        ols_in_bounds = []
    for obj_key in obj_arcs_above_cutoff:
        if return_dict:
            for fr,bl in arc_segment_in_interval(objs[obj_key],start,end):
                try:
                    ols_in_bounds[fr-ols_offset].append(ols[fr-ols_offset][bl])
                except:
                    pass
        else:
            ols_in_bounds.extend([ols[fr-ols_offset][bl] for fr,bl in arc_segment_in_interval(objs[obj_key],start,end)])
    return ols_in_bounds

def greedy_objs_filter(objs,ols,size_h,size_bins,fol_h,fol_bins,SHAPE):
    keep = []
    drop = []
    for o in sorted(objs,key=lambda x: score_object_arc(x,ols,size_h,size_bins,fol_h,fol_bins,SHAPE),reverse=True):
        if any([len(set([this_fr_num for this_fr_num,this_bl_num in o]).intersection(set([keep_fr_num for keep_fr_num,keep_bl_num in keep_obj]))) for keep_obj in keep]):
            drop.append(o)
        else:
            keep.append(o)
    return keep,drop

def greedy_objs_filter_from_values(obj_arcs,size_values,fol_values,size_h,size_bins,fol_h,fol_bins):
    #RETURNS SCORES AND OBJECTS
    keep = []
    drop = []
    keep_scores = []
    drop_scores = []
    for score,o in sorted([(score_object_arc_from_values(sv,fv,size_h,size_bins,fol_h,fol_bins),o_cand) for o_cand,sv,fv in zip(obj_arcs,size_values,fol_values)],reverse=True):
        if any([len(set([this_fr_num for this_fr_num,this_bl_num in o]).intersection(set([keep_fr_num for keep_fr_num,keep_bl_num in keep_obj]))) for keep_obj in keep]):
            drop.append(o)
            drop_scores.append(score)
        else:
            keep.append(o)
            keep_scores.append(score)
    return keep,drop,keep_scores,drop_scores
    
def mousemask_from_object_arcs(start,end,min_arc_score,ols,ols_offset,objs,objs_sizes,objs_fols,size_h, size_bins, fol_h, fol_bins,SHAPE):
    #replace with actual logic to take an interval definition (absolute) and objs
    # greedy filter objs, take keeps, get ols from interval and mask_from; sum and return
    included_ols = ols_in_interval(start,end,min_arc_score,ols,ols_offset,objs,objs_sizes,objs_fols,size_h, size_bins, fol_h, fol_bins)

    if len(included_ols) > 0:
        return reduce(lambda x,y:x+y, [mask_from_outline(p,SHAPE) for p in included_ols])
    else:
        return numpy.zeros(SHAPE,dtype=bool)

def chain_outlines_from_mask_old(mask,origin=(0,0),grow_by=0,preshrink=2,xybounds=None,border=50):
    '''given a mask and an origin, "hops" from blob to blob chaining an outline'''
    
    #newmask = mask.copy()
    
    #added 20100508 to address "hang" on analysis at edges of matrix
    if border > 0:
        ydim,xdim = mask.shape
        newmask = numpy.zeros((ydim+2*border,xdim+2*border))
        newmask[border:-border,border:-border] = mask
    else:
        newmask = mask.copy()
    
    if preshrink:
        newmask = shrink_mask(newmask,preshrink)
    #	if grow_by:
    #		grow_by += preshrink
    #if grow_by:
        newmask = grow_mask(newmask,preshrink)
    
    outlines = []
    termini = []
    start = origin
    thisiter = 0
    maxiter = 100
    while newmask.any():
        if thisiter > maxiter:
            print >> sys.stderr,'iteration passed maxiter (%s)' % maxiter
            break
        outlines.append(outline_from_mask(newmask,start,grow_by=grow_by))
        end = farthest_point(outlines[-1],start)
        termini.append([outlines[-1][0],end])
        blobmask = mask_from_outline(outlines[-1],newmask.shape)
        newmask = Util.subtract_mask(newmask,blobmask,0)
        thisiter += 1
    
    #need to reset all xys to reflect original mask
    resetouts = []
    resetterms = []
    for p in outlines:
        resetp = []
        for x,y in p:
            resetp.append((x-border,y-border))
        resetouts.append(resetp)
    for p in termini:
        resetp = []
        for x,y in p:
            resetp.append((x-border,y-border))
        resetterms.append(resetp)
    
    #return outlines,termini
    return resetouts,resetterms

def outline_from_polygon(poly):
    pd_poly = []
    for itr in range(len(poly)):
        i = itr-1
        m,b = line_fx_from_pts(map(float,poly[i]),poly[i+1])
        step = poly[i][0] < poly[i+1][0] and 1 or -1
        edge = [(x,m*x+b) for x in xrange(int(poly[i][0]),int(poly[i+1][0]),step)]
        pd_poly.extend(edge)

    return pd_poly

def distance_to_edge_mat(poly,SHAPE):
    mask = mask_from_outline(poly,SHAPE)
    dmat = numpy.zeros(mask.shape,dtype=float)
    for x in xrange(dmat.shape[0]):
        for y in xrange(dmat.shape[1]):
            if mask[x,y]:
                dmat[x,y] = min(distance_from_point(poly,(y,x)))
    return dmat                                    

def mask_from_outline(outline,shape):
    if len(outline) < 2:
        return numpy.zeros(shape,dtype=bool)

    img = Image.new('L', tuple(reversed(shape)), 0)
    ImageDraw.Draw(img).polygon(outline, outline=1, fill=1)
    return numpy.array(img).astype(bool)

def mask_from_outline_old(outline,shape):
    '''given dimensions (shape), generates a bool mask that is True inside shape outline (list of (x,y) coords)'''
    #print >> sys.stderr, "generate mask from outline:",outline
    xsorted = sorted(outline)
    newmask = numpy.zeros(shape,dtype=bool)
    newmask = newmask.transpose()
    
    while xsorted:
        drop = None
        top = xsorted.pop(0)
        last = top[1]
        while xsorted and xsorted[0][0] == top[0] and xsorted[0][1] == last+1:
            drop = xsorted.pop(0)
            last = drop[1]
        if xsorted and xsorted[0][0] == top[0]:				
            bot = xsorted.pop(0)
            while xsorted and xsorted[0][0] == bot[0] and xsorted[0][1] == bot[1]+1:
                bot = xsorted.pop(0)
            try:
                newmask[top[0]][top[1]:bot[1]+1] = True
            except IndexError:
                pass
        elif drop:
            try:
                newmask[top[0]][top[1]:drop[1]+1] = True
            except IndexError:
                pass
        else:
            #print top
            try:
                newmask[top[0]][top[1]] = True
            except IndexError:
                pass
        
    return newmask.transpose()

def subtract_outline(ol,frame,replace_with = None):
    if replace_with is None:
        replace_with = frame.min()
    sub_mask = mask_from_outline(ol,frame.shape)
    # DUMB:
    #for i,r in enumerate(frame):
    #	for j,v in enumerate(r):
    #		if sub_mask[i,j]: frame[i,j] = replace_with
    #SMART:
    frame[sub_mask] = replace_with

def subtract_masks(frame,masks,replace_with=None):
    if replace_with is None:
        replace_with = numpy.mean(frame[:50,:50])
    sum_masks = reduce(lambda x,y: x+y, masks)
    frame_copy = frame.copy()
    frame_copy[sum_masks] = replace_with
    return frame_copy

def size_of_polygon(p):
    '''given a polygon outline (list of points) returns the size in pixels of the encompassed area'''
    x,y = Util.dezip(p)
    xmax = max(x)
    ymax = max(y)
    mask = mask_from_outline(p,(ymax+2,xmax+2))
    return len(mask.flat[mask.flat>0])

def center_of_polygon(p):
    '''given a polygon outline, returns the center point (farthest from nearest edge)'''
    x,y = Util.dezip(p)
    xmax = max(x)
    ymax = max(y)
    mask = mask_from_outline(p,(ymax+2,xmax+2))
    pts = points_from_mask(mask)
    return max([(hypotenuse(closest_point(p,pt),pt),pt) for pt in pts])[1]

def calc_coordMat(SHAPE):
    return numpy.array([[tuple((i,j)) for j in range(SHAPE[1])] for i in range(SHAPE[0])])

def centroid(ol,SHAPE,coordMat=None):
    '''
    given a list of points <ol> and 2-tuple SHAPE return centroid of ol
    if coordMat is None, compute from SHAPE (significantly slower)
    to calculate coordMat independently, see calc_coordMat()
    '''

    if coordMat is None:
        coordMat = calc_coordMat(SHAPE)
    m = mask_from_outline(ol,SHAPE)
    return tuple(reversed(map(numpy.mean,zip(*coordMat[m])))) 

def centroids_from_mouseols_tar(tarf,centfile=None):
    import tarfile
    if centfile and os.path.exists(centfile):
        centroids = numpy.load(centfile)
        print >> sys.stderr, 'centroid output %s exists; load %s frames from file' % (centfile,len(centroids))
        return centroids
    
    tarh = tarfile.open(tarf)
    tarn = tarh.getnames()
    SHAPE = eval(open(os.path.dirname(tarf)+'/SHAPE').read())
    cm = calc_coordMat(SHAPE)
    mice = Util.tar2obj(tarn[0],tarh)
    hsl = len(mice)
    centroids = numpy.zeros((hsl*len(tarn),2),dtype=int)
    print >> sys.stderr, 'load %s half-segments of length %s; array dimensions: %s' % (len(tarn),hsl,centroids.shape)
    for i,n in enumerate(tarn):
        print >> sys.stderr, '\r%s/%s' % (i,len(tarn)),
        for j,oll in enumerate(Util.tar2obj(n,tarh)):
            if len(oll) == 1:
                x,y = centroid(oll[0],SHAPE,cm)
                centroids[(i*hsl)+j,0] = int(x)
                centroids[(i*hsl)+j,1] = int(y)

    if centfile:
        numpy.save(open(centfile,'wb'),centroids)
        
    return centroids
                                                        
def centroids_to_mean_rate(centroids,binwidth):
    '''binwidth = half segment length, eg'''
    allmeans = []
    alllens = []
    thisarc = []
    last = (0,0)
    dist_by_bin = []
    for i,(x,y) in enumerate(centroids):
        if i%int(binwidth) == 0:
            if thisarc:
                allmeans.append(numpy.mean(thisarc))
                alllens.append(len(thisarc))
                thisarc = []
            m = numpy.mean(allmeans)
            dist_by_bin.append(m>0 and m or None) #,numpy.mean(alllens),len(allmeans),sum(alllens)
            allmeans = []
            alllens = []
        if last != (0,0) and (x,y) != (0,0):
            thisarc.append(hypotenuse(last,(x,y)))
        else:
            if thisarc:
                allmeans.append(numpy.mean(thisarc))
                alllens.append(len(thisarc))
                thisarc = []
        last = (x,y)
    return dist_by_bin[1:]

def is_edge(mask,point,xybounds=None):
    '''returns true if the value (x,y) (i.e. mask[y,x]) is true, and an adjacent cell is false'''
    if mask[point[1],point[0]]:
        return not all(get_adjacent_values(mask,point))
    else:
        raise ValueError, '%s in mask is False (this is bad)' % (point,)

def closest_edge(mask,point):
    '''finds the nearest edge (true cell adjacent to false cell) to point'''
    cp = closest_point(points_from_mask(mask),point)
    if is_edge(mask,cp):
        return cp
    else:
        return (point[0],numpy.arange(len(mask[:,point[0]]))[mask[:,point[0]]].min())
