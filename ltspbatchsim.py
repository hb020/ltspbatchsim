import numpy as np
import spicelib
from spicelib.simulators.ltspice_simulator import LTspice
from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter, AutoMinorLocator, FormatStrFormatter
import os
import shutil
import glob
import json
import argparse
import math
import fnmatch
import sys
import logging
import subprocess

# Run a series of simulations on a model, with varying variable values. Puts the result in one single png (per job).

outdir = "./batchsim/"

class mySimulator(LTspice):
    # one could force the paths here
    # spice_exe = []
    # process_name = None
    pass


default_config_file = "ltspbatchsim.json"
CONFIG = {}

additional_files = ["*.lib", "*.LIB", "*.asy", "*.ASY"]

# temporary filenames (actually, the base names)
tmp_filenames = []


def delfiles(pattern):
    for p in glob.glob(pattern, recursive=False):
        if os.path.isfile(p):
            os.remove(p)


def tmp_filenames_register(fname: str):
    global tmp_filenames
    tmp_filenames.append(fname)
    

def tmp_filenames_cleanup(keep_nets=False, keep_logs=False, keep_raw=False):
    for fname in tmp_filenames:
        if not keep_nets:
            delfiles(f"{fname}*.net")
        if not keep_logs:
            delfiles(f"{fname}*.log")
        if not keep_raw:
            delfiles(f"{fname}*.raw")
    
    
# format the graph axis
def format_axis_transient(ax, size, offset_time_s, duration_time_s, ylabel, jobname):
    # print(f"size: {size}, offset_time_s: {offset_time_s}, duration_time_s: {duration_time_s}")
    ax.set_xscale("linear")
    formatter0 = EngFormatter(unit='s')
    ax.xaxis.set_major_formatter(formatter0)
    ax.xaxis.set_tick_params('both')
    ax.set_xlim(left=0, right=duration_time_s)
    if size > 0:
        # nice formatting, on low number of sections
        # major on every 0.5 Xs:
        ax.set_xticks(np.arange(0, duration_time_s * 1.0001, duration_time_s / (2 * size)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        # remove half of the x labels
        xlabels = ax.xaxis.get_ticklabels(minor=False)
        i = 1
        while i < len(xlabels):
            xlabels[i] = ''
            i += 2
        ax.xaxis.set_ticklabels(xlabels, minor=False)

    if offset_time_s == 0:
        ax.set_xlabel("time (s)")
    else:
        if offset_time_s >= 1:
            s = f"{offset_time_s} s"
        else:
            if offset_time_s >= 0.001:
                s = f"{offset_time_s * 1000:.6g} ms"
            else:
                s = f"{offset_time_s * 1000000:.6g} µs"
        ax.set_xlabel(f"time (s), offset by {s}")

    ax.set_ylabel(ylabel)
    ax.set_yscale("linear")
    formatter1 = EngFormatter()
    ax.yaxis.set_major_formatter(formatter1)    

    ax.grid(visible=True, which="both", axis="both")
    ax.grid(which="minor", axis="both", alpha=0.5)
    if jobname:
        ax.set_title(jobname)
    ax.legend()    


def format_axis_ac(ax, is_gain, single_bode, x_scale, start_freq, end_freq, ylabel, jobname):
    if is_gain or not single_bode:
        if x_scale == "lin":
            ax.set_xscale("linear")
        else:
            ax.set_xscale("log")
        formatter0 = EngFormatter(unit='Hz')
        ax.xaxis.set_major_formatter(formatter0)
        ax.xaxis.set_tick_params('both')
        ax.set_xlim(left=start_freq, right=end_freq)
        ax.set_xlabel("Frequency")
    
    ext = ""
    if is_gain:
        ext = "gain"
        unit = "%g dB"
    else:
        unit = "%g °"
        ext = "phase"
        if single_bode:
            ext = ext + " (dashed)"
            
    ax.set_ylabel(f"{ylabel} - {ext}")
    formatter1 = FormatStrFormatter(unit)
    ax.yaxis.set_major_formatter(formatter1)

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    
    if is_gain or not single_bode:
        ax.grid(visible=True, which="both", axis="both")
        ax.grid(which="minor", axis="both", alpha=0.5)    
        if jobname:
            ax.set_title(jobname)
        ax.legend()    


def str2float(s: str):
    if s is None:
        return math.nan
    s = s.strip()
    if len(s) == 0:
        return math.nan
    s = s.lower()
    mult = 1
    if s.endswith("f"):
        mult = 1e-15
        s = s[:-1]
    elif s.endswith("p"):
        mult = 1e-12
        s = s[:-1]        
    elif s.endswith("n"):
        mult = 1e-9
        s = s[:-1]        
    elif s.endswith("u"):
        mult = 1e-6
        s = s[:-1]        
    elif s.endswith("m"):
        mult = 1e-3
        s = s[:-1]        
    elif s.endswith("k"):
        mult = 1e3
        s = s[:-1]        
    elif s.endswith("meg"):
        mult = 1e6
        s = s[:-3]        
    elif s.endswith("mega"):
        mult = 1e6
        s = s[:-4]
    elif s.endswith("g"):
        mult = 1e9
        s = s[:-1]        
    elif s.endswith("t"):
        mult = 1e12
        s = s[:-1]
    try:
        r = float(s)
    except ValueError:
        return math.nan
    return r * mult


# will not do smaller than nano seconds
def time2nsecs(t: str):
    f = str2float(t)
    if math.isnan(f):
        return -1
    return round(1000000000.0 * f)


def freq2Hz(t: str):
    f = str2float(t)
    return f


def run_analysis(job, jobnr, nrjobs, take_me, showplot=True, model_fname="", defaultac="", defaulttransients=[], defaultlabels=[], 
                 single_bode=False, use_asc=False, dense=False, defaultaltsolver=False):
    nrcols = 0
    nrrows = 0
    
    jobname = job["name"]
    ac_analysis = False
    if "op" in job:
        op = job["op"].lower()
        if op.startswith('ac'):
            ac_analysis = True
            if op == 'ac1':
                single_bode = True
            if op == 'ac2':
                single_bode = False
    
    alt_solver = defaultaltsolver
    if "alt" in job:
        alt_solver = job["alt"]
    
    if not take_me:
        logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Skipping")
        return
    else:
        logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", {'AC' if ac_analysis else 'Transient'} analysis.")         
    
    ylabels = []
    if "ylabel" in job:
        ylabels.append(job["ylabel"])
    elif "ylabels" in job:
        ylabels = job["ylabels"]
    else:
        ylabels = defaultlabels
    nrrows = len(ylabels)
    
    # set up graph columns
    column_definitions = []
    
    if ac_analysis:
        # AC graph setup        
        ac = ""
        if "ac" in job:
            ac = job["ac"]
        else:
            ac = defaultac        
        ac = ac.strip()
        if len(ac) == 0:
            logger.error("You need to specify an AC analysis string via 'ac'.")
            return
        
        # ac format is the spice format for .AC: 
        # "<oct, dec, lin> <Nsteps> <StartFreq> <EndFreq>"
        start_freq = math.nan
        end_freq = math.nan
        method = ""
        tparts = [x.strip() for x in ac.split(' ')]
        if len(tparts) == 4:
            method = tparts[0].lower()
            if method in ['oct', 'dec', 'lin']:
                start_freq = freq2Hz(tparts[2])
                end_freq = freq2Hz(tparts[3])
        
        if math.isnan(start_freq) or math.isnan(end_freq):
            logger.error(f"Unsupported transient AC analysis format \"{ac}\".")
            logger.error("  Format must be \"<oct, dec, lin> <Nsteps> <StartFreq> <EndFreq>\".")
            return
        
        # gain and phase in separate columns, it would be too dense otherwise
        column_definitions.append({"name": "gain", "start_freq": start_freq, "end_freq": end_freq, "method": method})
        if not single_bode:
            column_definitions.append({"name": "phase", "start_freq": start_freq, "end_freq": end_freq, "method": method})
    else:
        # transient graphs setup
        transients = []
        if "transient" in job:
            transients.append(job["transient"])
        if "transients" in job:
            transients = job["transients"]
        else:
            transients = defaulttransients
        
        maxtime_nsecs = 0
        # transient time format is a subset of the spice format for .TRAN: 
        # "<Tstop>"
        # "0 <Tstop> Tstart" (=> Tstep must be 0, dTmax is not used)
        for t in transients:
            startrecording_nsecs = 0
            endrecording_nsecs = 0
            visible_nsecs = 0
            # get the "dense" designator, and strip off if present
            t_dense = False
            t = t.lower().strip()
            if t.endswith(" d"):
                t_dense = True
                t = t[:-2]
            # strip the string in sections, take out repeated spaces
            tparts = [x.strip() for x in t.split(' ') if x]
            # interpret the numerical parts
            if len(tparts) == 1:
                startrecording_nsecs = 0
                endrecording_nsecs = time2nsecs(tparts[0])
                visible_nsecs = endrecording_nsecs
            elif len(tparts) == 2:
                startrecording_nsecs = time2nsecs(tparts[1])
                endrecording_nsecs = time2nsecs(tparts[0])
                visible_nsecs = endrecording_nsecs - startrecording_nsecs
            elif len(tparts) == 3 and tparts[0] == "0":
                startrecording_nsecs = time2nsecs(tparts[2])
                endrecording_nsecs = time2nsecs(tparts[1])
                visible_nsecs = endrecording_nsecs - startrecording_nsecs
            else:
                startrecording_nsecs = -1  # signal error
            
            if startrecording_nsecs < 0 or endrecording_nsecs < 0:
                logger.error(f"Unsupported transient time format \"{t}\".")
                logger.error("  Format must be either \"Tstop\" or \"Tstop Tstart\" or \"0 <Tstop> Tstart\". (First and last forms are like LTSpice .TRAN)")
                return

            if visible_nsecs <= 0:
                logger.error(f"Unsupported transient time format \"{t}\", duration time cannot be 0 or negative.")
                logger.error("  Format must be either \"Tstop\" or \"Tstop Tstart\" or \"0 Tstop Tstart\". (First and last forms are like LTSpice .TRAN)")
                return
            
            column_definitions.append({"name": t,
                                       "startrecording_nsecs": startrecording_nsecs, 
                                       "endrecording_nsecs": endrecording_nsecs, 
                                       "visible_nsecs": visible_nsecs,
                                       "dense": t_dense})
            if maxtime_nsecs < endrecording_nsecs:
                maxtime_nsecs = endrecording_nsecs
        # add some more, sometimes I get stange effects at the end
        maxtime_nsecs = int(maxtime_nsecs * 1.1)
                
    nrcols = len(column_definitions)
    # print(column_definitions)
                 
    if nrcols == 0 or nrrows == 0:
        logger.error(f"No plots defined for {jobname}")
        return

    # set the graph width, every column has the same width (as is the easiest with matplot)
    xsize = 10
    if not ac_analysis:
        # adapt the width so that I see some detail
        for c in column_definitions:
            nicenumber = 0
            t = c["visible_nsecs"]  # is integer
            wantedsize = 10
            while nicenumber == 0:
                if t < 40:
                    # adapt the size a bit
                    # nice small number 
                    wantedsize = max(10, t / 1.5)
                    nicenumber = t
                else:
                    if (t % 10 == 0):
                        t = int(t / 10)
                    else:
                        break

            if nicenumber == 1:
                nicenumber = 10
            c["niceformat"] = nicenumber

            if wantedsize > xsize:
                xsize = wantedsize
    
    # create the rows and columns in the graph
    # differents ylabels are on rows (y axis of grid)
    # time graphs are in columns (x axis of grid)
    fig, ax = plt.subplots(nrrows, nrcols, dpi=150, figsize=(xsize * nrcols, (4.8 * nrrows)), squeeze=False)        
    fig.set_layout_engine('tight')
    if ac_analysis and single_bode:
        # ax is a numpy.ndarray. And that is messy to incrementally expand
        axnew = []
        for row in range(0, nrrows):
            axnew.append([ax[row][0], ax[row][0].twinx()])  # create a second Axes that shares the same x-axis
            # do not change column_definitions, as I do not have a second column.
        # overwrite ax 
        ax = axnew
        # no forced dashing when in single bode, as I already do that.
        dense = False

    # now prepare the spice model
    if use_asc:
        netlist = spicelib.AscEditor(f"./{model_fname}")  # Open the Spice Model, and creates the tailored .asc file
    else:
        netlist = spicelib.SpiceEditor(f"./{model_fname}")  # Open the Spice Model, and creates the tailored .net file

    basename = os.path.join(outdir, jobname.replace(' ', '_'))
    tmp_filenames_register(basename)
       
    traceidx = 0
    nrtraces = len(job["traces"])
    for tracename in job["traces"]:
        if traceidx != 0:
            # no need to reset if I never changed it
            netlist.reset_netlist()
        
        # Set simulation instructions
        netlist.remove_Xinstruction(".*\\.tran.*")  # is case insensitive
        netlist.remove_Xinstruction(".*\\.ac.*")  # is case insensitive
        if ac_analysis:
            netlist.add_instruction(f".ac {ac}")
        else:
            netlist.add_instruction(f".tran {maxtime_nsecs}n")        
        
        # create "dense" line style. Decide later on if it you take it or not 
        if (nrtraces == 1) or (traceidx == 0):
            dense_linestyle = 'solid'
        else:
            dense_linestyle = (traceidx - 1, (nrtraces - 1, nrtraces - 1))
        traceidx += 1
        
        # get a good trace name
        safe_tracename = tracename.replace(",", "").replace(" ", "_")
        if use_asc:
            netlistfile = f"{basename}_{safe_tracename}.asc"
        else:
            netlistfile = f"{basename}_{safe_tracename}.net"
        rawfile = f"{basename}_{safe_tracename}.raw"
        processlogfile = f"{basename}_{safe_tracename}_process.log"
        
        for s in ["commondefs", "tracedefs"]:
            if s in job: 
                for k in job[s]:
                    d = {}
                    if s == "tracedefs":
                        if isinstance(k, dict):
                            for kk, kv in k.items():
                                if "{name}" in kv:
                                    kv = kv.replace("{name}", tracename)
                                elif "{name" in kv:
                                    if ',' in tracename:
                                        tracename_parts = [x.strip() for x in tracename.split(',')]
                                    else:
                                        tracename_parts = [x.strip() for x in tracename.split(' ')]
                                    i = 1
                                    for p in tracename_parts:
                                        kv = kv.replace(f"{{name{i}}}", p)
                                        i += 1
                                d[kk] = kv
                        else:
                            if "{name}" in k:
                                k = k.replace("{name}", tracename)
                            elif "{name" in k:
                                tracename_parts = [x.strip() for x in tracename.split(',')]
                                i = 1
                                for p in tracename_parts:
                                    k = k.replace(f"{{name{i}}}", p)
                                    i += 1
                            d.update(CONFIG["defs"][k])
                    else:
                        if isinstance(k, dict):
                            d.update(k)
                        else:                        
                            d.update(CONFIG["defs"][k])
                            
                    for kk, kv in d.items():
                        if use_asc and kk.startswith("X"):
                            # remove the X when using spicelib.AscEditor
                            kk = kk[1:]
                        logger.debug(f"set_component_value({kk}:{kv})")                                
                        netlist.set_component_value(kk, kv)
                        if use_asc:
                            # wipe Value2 if in asc
                            netlist.set_component_parameters(kk, **{"Value2": ""})                            
                                            
        netlist.save_netlist(netlistfile)
        
        logger.info(f"Job: {jobname}: Trace '{tracename}'")
        opts = []
        if alt_solver:
            opts.append('-alt')
        else:
            opts.append('-norm')
            
        with open(processlogfile, "w") as outfile:
            mySimulator.run(netlistfile, opts, timeout=None, stdout=outfile, stderr=subprocess.STDOUT)
        
        LTR = spicelib.RawRead(rawfile)
        # LTR.to_excel("out.xlsx")
        for row in range(0, nrrows):
            y = LTR.get_trace(ylabels[row])
            if ac_analysis:
                xlabel = "frequency"
            else:
                xlabel = "time"
            x = LTR.get_trace(xlabel)
            steps = LTR.get_steps()
            
            for step in range(len(steps)):
                t = x.get_wave(step)
                v = y.get_wave(step)
                for col in range(0, nrcols):
                    c = column_definitions[col]
                    t_dense = dense
                    if ("dense" in c) and c["dense"]:
                        t_dense = True
                    if t_dense:
                        linestyle = dense_linestyle
                    else:
                        linestyle = 'solid'

                    if ac_analysis:
                        # I always have 2 axis, no matter what nrcols says. 
                        # so skip column 1, and do both at the same time
                        # this construction prevents extra code in case of single_bode
                        if col == 0:                            
                            ax[row][0].plot(np.real(t), 20.0 * np.log10(np.abs(v)), label=tracename, linestyle=linestyle)
                            if single_bode:
                                ax[row][1].plot(np.real(t), np.degrees(np.angle(v)), label=tracename, linestyle='dashed')
                            else:
                                ax[row][1].plot(np.real(t), np.degrees(np.angle(v)), label=tracename, linestyle=linestyle)   
                    else:
                        startrecording = c["startrecording_nsecs"] / 1000000000.0
                        endrecording = c["endrecording_nsecs"] / 1000000000.0
                        startidx = None
                        endidx = None
                        for sample in range(len(t)):
                            if startidx is None:
                                if t[sample] >= startrecording:
                                    startidx = max(0, sample - 1)
                            if endidx is None:
                                if t[sample] > endrecording:
                                    endidx = min(len(t), sample + 1)
                        if startidx is None:
                            startidx = 0
                        if endidx is None:
                            endidx = len(t)
                        ax[row][col].plot(t[startidx:endidx] - startrecording, v[startidx:endidx], label=tracename, linestyle=linestyle)

    logger.info(f"Job: {jobname}: Creating graph.")
    for row in range(0, nrrows):
        for col in range(0, nrcols):
            if row == 0 and col == 0:
                title = jobname
            else:
                title = None
            c = column_definitions[col]                
            if ac_analysis:
                start_freq = c["start_freq"]
                end_freq = c["end_freq"]
                x_scale = c["method"]
                # I always have 2 axis, no matter what nrcols says. 
                # so skip column 1, and do both at the same time
                # this construction prevents extra code in case of single_bode
                if col == 0:
                    format_axis_ac(ax[row][0], True, single_bode, x_scale, start_freq, end_freq, ylabels[row], title)
                    format_axis_ac(ax[row][1], False, single_bode, x_scale, start_freq, end_freq, ylabels[row], title)
            else:
                startrecording_s = c["startrecording_nsecs"] / 1000000000.0
                visible_s = c["visible_nsecs"] / 1000000000.0
                size = c["niceformat"]
                format_axis_transient(ax[row][col], size, startrecording_s, visible_s, ylabels[row], title)

    imagefile = f"{basename}.png"
    fig.savefig(fname=imagefile, dpi=300)
    logger.info(f"Job: {jobname}: The results are in the file {imagefile}")
    if showplot:
        fig.show()


def prepare():
    for cf in additional_files:
        for f in glob.glob(cf):
            shutil.copyfile(f"./{f}", f"{outdir}{f}")


def load_config(config_file):
    global CONFIG
    logger.info(f"Using config file \"{config_file}\".")
    f = open(config_file)
    CONFIG = json.load(f)


if __name__ == "__main__":

    # get path from LTSpice, as determined by spicelib
    if len(mySimulator.spice_exe) == 1:
        winepath = None
        ltspice_path = mySimulator.spice_exe[0]
    else:
        winepath = mySimulator.spice_exe[0]
        ltspice_path = mySimulator.spice_exe[1]
                    
    parser = argparse.ArgumentParser(description="Runs one or more LTSpice simulations based on config from a json file. "
                                     "Will use LTSpice installed under wine.")
    parser.add_argument('config_file', default=default_config_file, help=f"Name of the config json file. Default: '{default_config_file}'") 
    parser.add_argument('job_name', default="", nargs="*", help="Name of the job(s) to run. If left empty: all jobs will be run. Wildcards can be used, but please escape the * and ? to avoid shell expansion. Example of good use in shell: \"test_OPA189\\*\", which will be passed on to this program as \"test_OPA189*\".") 
    parser.add_argument('--ltspicepath', default=ltspice_path, help=f"Path of ltspice. Default: '{ltspice_path}'")
    if sys.platform == 'linux' or sys.platform == 'darwin':
        parser.add_argument('--winepath', default=winepath, help=f"Path of wine, if used. Default: '{winepath}'")
    parser.add_argument('--outdir', default=outdir, help=f"Output directory for the graphs, also work directory. Default: '{outdir}'")
    parser.add_argument('--use_asc', default=False, action='store_true', help="Run the simulations as usual, but do that using .asc files. This is somewhat slower, but can be useful for diving into problems detected with the simulations, as it keeps the .asc files after the simulations.")
    parser.add_argument('--keep_nets', default=False, action='store_true', help="After the runs, keep the netlists.")
    parser.add_argument('--keep_logs', default=False, action='store_true', help="After the runs, keep the spice run logs.")
    parser.add_argument('--keep_raw', default=False, action='store_true', help="After the runs, keep the .raw files.")
    parser.add_argument('--single_bode', default=False, action='store_true', help="Keep AC analysis bode plots in the same graph, instead of having gain and phase in separate columns.")
    parser.add_argument('--dense', default=False, action='store_true', help="Use this if the graph is dense. It will dash the lines, making distinction easier. Not used with '--single_bode'")
    parser.add_argument('-v', '--verbose', default=logging.INFO, help="Be verbose", action="store_const", dest="loglevel", const=logging.DEBUG)
    parser.add_argument('--log', default=False, action='store_true', help="Log to file, not to console. If set, will log to \"{config_file}.log\", in append mode.")
    
    # use_asc is not preferred, as spicelib has some limitations in the spicelib.AscEditor.
    
    args = parser.parse_args()
    
    # this all is before the logging setup, so use print in case of problems
    outdir = args.outdir
    if not (outdir.endswith('\\') or outdir.endswith('/')):
        print(f"ERROR: Output directory \"{outdir}\" is not terminated with a slash. Please do so.")
        exit(1)
    if not os.path.isdir(outdir):
        print(f"ERROR: Output directory \"{outdir}\" cannot be found. Please specify a valid path via --outdir.")
        exit(1)
    
    if not os.path.isfile(args.config_file):
        print(f"ERROR: Config file \"{args.config_file}\" cannot be found. Please specify a valid path.")
        exit(1)
        
    if args.log:
        fname, fext = os.path.splitext(args.config_file) 
        logfile = os.path.join(outdir, fname + ".log")
        print(f"logging to \"{logfile}\"")
        # configure logging
        logging.basicConfig(filename=logfile, level=args.loglevel)
    else:
        logging.basicConfig(level=args.loglevel)
        
    logger = logging.getLogger(__name__)
        
    if args.loglevel == logging.DEBUG:
        spicelib.set_log_level(logging.INFO)
    else:
        spicelib.set_log_level(logging.WARNING)
                
    ltspice_path = args.ltspicepath
    if sys.platform == 'linux' or sys.platform == 'darwin':
        winepath = args.winepath
        if len(winepath) == 0:
            winepath = None
    else:
        winepath = None
        
    if not os.path.isfile(ltspice_path):
        logger.error(f"ltspice is not found under \"{ltspice_path}\". Please specify a valid path via --ltspicepath.")
        exit(1)
        
    if winepath:
        if not shutil.which(winepath):
            logger.error(f"wine is not found under \"{winepath}\". Please specify a valid path via --winepath.")
            exit(1)            

    new_spice_exe = [ltspice_path]
    if winepath:
        new_spice_exe = [winepath, ltspice_path]
    if mySimulator.spice_exe != new_spice_exe:
        mySimulator.spice_exe = new_spice_exe
        mySimulator.process_name = None  # let the lib find out the process name
                    
    load_config(args.config_file)
    prepare()
    
    my_jobs = args.job_name
    defaulttransients = []
    defaultac = ""
    defaultlabels = []
    defaultaltsolver = False
    if "transients" in CONFIG:
        defaulttransients = CONFIG["transients"]
    if "ac" in CONFIG:
        defaultac = CONFIG["ac"]
    if "ylabels" in CONFIG:
        defaultlabels = CONFIG["ylabels"]
    if "alt" in CONFIG:
        defaultaltsolver = CONFIG["alt"]
        
    model_fname = CONFIG["model"]
    if not os.path.isfile(model_fname):
        logger.error(f"Cannot find model file \"{model_fname}\".")
        exit(1)
        
    model_root, model_ext = os.path.splitext(model_fname)
    if args.use_asc:
        if model_ext.lower() != ".asc":
            logger.error(f"You must provide a .asc file as model when you use --use_asc. The name \"{model_fname}\"is invalid.")
            exit(1)
    else:
        if model_ext.lower() == ".asc":
            # convert to .net file
            logger.info(f"Creating the netlist for the schema \"{model_fname}\"")
            # make sure the log file goes to outdir
            processlogfile = os.path.join(outdir, f"{model_root}_create_netlist.log")
            with open(processlogfile, "w") as outfile:
                model_fname = mySimulator.create_netlist(model_fname, stdout=outfile, stderr=subprocess.STDOUT)

            if not os.path.isfile(model_fname):
                logger.error(f"Cannot find netlist file \"{model_fname}\", something went wrong. See file {processlogfile}.")
                exit(1)

    logger.info(f"Running simulations, on netlist \"{model_fname}\".")
            
    nrjobs = len(CONFIG["run"])
    jobnr = 0
    for job in CONFIG["run"]:
        jobnr += 1
        # look if I need to run this job
        take_me = True
        if len(my_jobs) > 0:
            take_me = False
            for j in my_jobs:
                j = j.lower()
                if '*' in j or '?' in j:
                    if fnmatch.fnmatch(job["name"], j):
                        take_me = True
                        break                        
                else:
                    if job["name"].lower() == j:
                        take_me = True
                        break
                
        run_analysis(job, 
                     jobnr, nrjobs, take_me, 
                     showplot=True, 
                     model_fname=model_fname,
                     defaultac=defaultac,
                     defaulttransients=defaulttransients,
                     defaultlabels=defaultlabels,
                     single_bode=args.single_bode,
                     use_asc=args.use_asc,
                     dense=args.dense,
                     defaultaltsolver=defaultaltsolver
                     )
    tmp_filenames_cleanup(args.keep_nets, args.keep_logs, args.keep_raw)
