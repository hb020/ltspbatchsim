import numpy as np
from PyLTSpice import SpiceEditor, RawRead
from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter, AutoMinorLocator, FormatStrFormatter
import os
import subprocess
import shutil
import glob
import json
import argparse
import math


# Run a series of simulations on a model, with varying variable values. Puts the result in one single png (per job).

# This does NOT use the MacOSX LTSpice, as the command line parameters are incomplete. It uses wine.
# I do NOT use any PyLTSpice runner, as it has problems with long command lines.

default_config_file = "ltspbatchsim.json"
ltspice_path = os.path.expanduser(f"~/.wine/drive_c{os.path.expanduser('~')}/AppData/Local/Programs/ADI/LTspice/LTspice.exe")
CONFIG = {}

additional_files = ["*.lib", "*.LIB", "*.asy", "*.ASY"]

outdir = "./batchsim/"

# djerickson:
# Irange  RIsense  C5
# 1uA     5M      100p
# 10uA    560k    370p
# 100uA   50k     370p
# 1mA     5k      370p
# 10mA    500     100p
# 100mA   50      100p (0.5W)

# Load,   120V  12V   
# Irange  Rload   Rload (max)
# 1uA     120M    12M
# 10uA    12M     1M2
# 100uA   1M2     120k
# 1mA     120k    12k
# 10mA    12k     1k2
# 100mA   1k2     120


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
    ax.set_ylabel(ylabel)
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


def time2usecs(t: str):
    f = str2float(t)
    if math.isnan(f):
        return -1
    return round(1000000.0 * f)


def freq2Hz(t: str):
    f = str2float(t)
    return f


def run_analysis(job, showplot=True, model_fname="", defaultac="", defaulttransients=[], defaultlabels=[], single_bode=False):
    nrcols = 0
    nrrows = 0
    
    jobname = job["name"]
    ac_analysis = False
    if "op" in job:
        if job["op"].lower() == 'ac':
            ac_analysis = True
    
    print(f"Job: {jobname}, {"AC" if ac_analysis else "Transient"} analysis.")         
    
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
            print("ERR: You need to specify an AC analysis string via 'ac'.")
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
            print(f"ERR: unsupported transient AC analysis format \"{ac}\".")
            print("  Format must be \"<oct, dec, lin> <Nsteps> <StartFreq> <EndFreq>\".")
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
        
        maxtime_usecs = 0
        # transient time format is a subset of the spice format for .TRAN: 
        # "<Tstop>"
        # "0 <Tstop> Tstart" (=> Tstep must be 0, dTmax is not used)
        for t in transients:
            startrecording_usecs = 0
            endrecording_usecs = 0
            visible_usecs = 0
            tparts = [x.strip() for x in t.split(' ')]
            if len(tparts) == 1:
                startrecording_usecs = 0
                endrecording_usecs = time2usecs(tparts[0])
                visible_usecs = endrecording_usecs
            elif len(tparts) == 2:
                startrecording_usecs = time2usecs(tparts[1])
                endrecording_usecs = time2usecs(tparts[0])
                visible_usecs = endrecording_usecs - startrecording_usecs
            elif len(tparts) == 3 and tparts[0] == "0":
                startrecording_usecs = time2usecs(tparts[2])
                endrecording_usecs = time2usecs(tparts[1])
                visible_usecs = endrecording_usecs - startrecording_usecs
            else:
                startrecording_usecs = -1  # signal error
            
            if startrecording_usecs < 0 or endrecording_usecs < 0:
                print(f"ERR: unsupported transient time format \"{t}\".")
                print("  Format must be either \"<Tstop>\" or \"<Tstop> Tstart\" or \"0 <Tstop> Tstart\". (First and last forms are like LTSpice .TRAN)")
                return
            
            column_definitions.append({"name": t,
                                       "startrecording_usecs": startrecording_usecs, 
                                       "endrecording_usecs": endrecording_usecs, 
                                       "visible_usecs": visible_usecs})
            if maxtime_usecs < endrecording_usecs:
                maxtime_usecs = endrecording_usecs
                
    nrcols = len(column_definitions)
    # print(column_definitions)
                 
    if nrcols == 0 or nrrows == 0:
        print(f"ERR: no plots defined for {jobname}")
        return

    # set the graph width, every column has the same width (as is the easiest with matplot)
    xsize = 10
    if not ac_analysis:
        # adapt the width so that I see some detail
        for c in column_definitions:
            nicenumber = 0
            t = c["visible_usecs"]  # is integer
            wantedsize = 10
            if t < 50:
                # nice small number of usecs
                wantedsize = max(10, t / 1.5)
                nicenumber = t
            elif t % 1000 == 0:
                # multiple of msecs
                t = int(t / 1000)
                if t < 50:
                    # nice small number of msecs
                    wantedsize = max(10, t / 1.5)
                    nicenumber = t
                if t % 1000 == 0:
                    # multiple of secs
                    t = int(t / 1000)
                    if t < 50:
                        # nice small number of secs
                        wantedsize = max(10, t / 1.5)
                        nicenumber = t
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

    # now prepare the spice model
    netlist = SpiceEditor(f"./{model_fname}")  # Open the Spice Model, and creates the .net

    basename = f"{outdir}{jobname}".replace(' ', '_')
    # remove old instructions
    netlist.remove_Xinstruction("\\.tran.*")  # is case insensitive
    netlist.remove_Xinstruction("\\.ac.*")  # is case insensitive
    if ac_analysis:
        netlist.add_instruction(f".ac {ac}")
    else:
        netlist.add_instruction(f".tran {maxtime_usecs}u")
        
    for tracename in job["traces"]:
        safe_tracename = tracename.replace(",", "").replace(" ", "_")
        netlistfile = f"{basename}_{safe_tracename}.net"
        rawfile = f"{basename}_{safe_tracename}.raw"
        processlogfile = f"{basename}_{safe_tracename}_process.log"
        
        for s in ["commondefs", "tracedefs"]:
            if s in job: 
                for k in job[s]:
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
                                netlist.set_component_value(kk, kv)                                
                        else:
                            if "{name}" in k:
                                k = k.replace("{name}", tracename)
                            elif "{name" in k:
                                tracename_parts = [x.strip() for x in tracename.split(',')]
                                i = 1
                                for p in tracename_parts:
                                    k = k.replace(f"{{name{i}}}", p)
                                    i += 1
                            d = CONFIG["defs"][k]
                            netlist.set_component_values(**d)
                    else:
                        if isinstance(k, dict):
                            for kk, kv in k.items():
                                netlist.set_component_value(kk, kv)
                        else:                        
                            d = CONFIG["defs"][k]
                            netlist.set_component_values(**d)                        
        
        netlist.save_netlist(netlistfile)
        
        print(f"Job: {jobname}: Trace '{tracename}'")
        spice_exe = ['wine', ltspice_path, '-alt', '-Run', '-b']
        with open(processlogfile, "w") as outfile:
            subprocess.run(spice_exe + [netlistfile], stdout=outfile, stderr=subprocess.STDOUT)
        
        LTR = RawRead(rawfile)
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
                    if ac_analysis:
                        # I always have 2 axis, no matter what nrcols says. 
                        # so skip column 1, and do both at the same time
                        # this construction prevents extra code in case of single_bode
                        if col == 0:
                            ax[row][0].plot(np.real(t), 20.0 * np.log10(np.abs(v)), label=tracename)
                            if single_bode:
                                ax[row][1].plot(np.real(t), np.degrees(np.angle(v)), label=tracename, linestyle='dashed')
                            else:
                                ax[row][1].plot(np.real(t), np.degrees(np.angle(v)), label=tracename)   
                    else:
                        startrecording = c["startrecording_usecs"] / 1000000.0
                        endrecording = c["endrecording_usecs"] / 1000000.0
                        if t.max() > startrecording and t.min() < endrecording:
                            ax[row][col].plot(t - startrecording, v, label=tracename)
    
    print(f"Job: {jobname}: Creating graph.")
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
                startrecording_s = c["startrecording_usecs"] / 1000000.0
                visible_s = c["visible_usecs"] / 1000000.0
                size = c["niceformat"]
                format_axis_transient(ax[row][col], size, startrecording_s, visible_s, ylabels[row], title)

    imagefile = f"{basename}.png"
    fig.savefig(fname=imagefile, dpi=300)
    print(f"Job: {jobname}: The results are in the file {imagefile}\n")
    if showplot:
        fig.show()


def prepare():
    for cf in additional_files:
        for f in glob.glob(cf):
            shutil.copyfile(f"./{f}", f"{outdir}{f}")
        

def delfiles(pattern):
    for p in glob.glob(f"{outdir}{pattern}", recursive=False):
        if os.path.isfile(p):
            os.remove(p)


def cleanup(keep_nets=False, keep_logs=False, keep_raw=False):
    if not keep_nets:
        delfiles("*.net")
    if not keep_logs:
        delfiles("*.log")
    if not keep_raw:
        delfiles("*.raw")


def load_config(config_file):
    global CONFIG
    print(f"Using config file {config_file}.")
    f = open(config_file)
    CONFIG = json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs one or more LTSpice simulations based on config from a json file. "
                                     "Will use LTSpice installed under wine.")
    parser.add_argument('config_file', default=default_config_file, help=f"Name of the config json file. Default: '{default_config_file}'") 
    parser.add_argument('job_name', default="", nargs="*", help="Name of the job(s). If left empty: all jobs will be run.") 
    parser.add_argument('--ltspicepath', default=ltspice_path, help=f"Path of ltspice. Default: '{ltspice_path}'")
    parser.add_argument('--outdir', default=outdir, help=f"Output directory for the graphs, also work directory. Default: '{outdir}'")
    parser.add_argument('--keep_nets', default=False, action='store_true', help="After the runs, keep the netlists.")
    parser.add_argument('--keep_logs', default=False, action='store_true', help="After the runs, keep the spice run logs.")
    parser.add_argument('--keep_raw', default=False, action='store_true', help="After the runs, keep the .raw files.")
    parser.add_argument('--single_bode', default=False, action='store_true', help="Keep AC analysis bode plots in the same graph, instead of having gain and phase in separate columns.")
    
    args = parser.parse_args()
    
    ltspice_path = args.ltspicepath
    if not os.path.isfile(ltspice_path):
        print(f"ERR: ltspice is not found under \"{ltspice_path}\". Please specify a valid path via --ltspicepath.")
        exit(1)
        
    outdir = args.outdir
    if not os.path.isdir(outdir):
        print(f"ERR: output directory \"{outdir}\" cannot be found. Please specify a valid path via --outdir.")
        exit(1)
    
    load_config(args.config_file)
    prepare()
    
    my_jobs = args.job_name
    defaulttransients = []
    defaultac = ""
    defaultlabels = []
    if "transients" in CONFIG:
        defaulttransients = CONFIG["transients"]
    if "ac" in CONFIG:
        defaultac = CONFIG["ac"]
    if "ylabels" in CONFIG:
        defaultlabels = CONFIG["ylabels"]
        
    model_fname = CONFIG["model"]
    if not os.path.isfile(model_fname):
        print(f"ERR: cannot find model file \"{model_fname}\".")
        exit(1)
        
    if model_fname.lower().endswith(".asc"):
        print(f"Extracting the netlist from the schema \"{model_fname}\"")
        spice_exe = ['wine', ltspice_path, '-netlist', ]
        subprocess.run(spice_exe + [model_fname], stderr=subprocess.STDOUT)
        model_fname = model_fname[:-4] + ".net"
        if not os.path.isfile(model_fname):
            print(f"ERR: cannot find netlist file \"{model_fname}\", something went wrong.")
            exit(1)

    print(f"\nRunning all the jobs, using the netlist \"{model_fname}\".")
            
    for job in CONFIG["run"]:
        if len(my_jobs) > 0:
            if job["name"] not in my_jobs:
                print(f"Skipping job '{job["name"]}'.")
                continue
        run_analysis(job, 
                     showplot=True, 
                     model_fname=model_fname,
                     defaultac=defaultac,
                     defaulttransients=defaulttransients,
                     defaultlabels=defaultlabels,
                     single_bode=args.single_bode
                     )
    cleanup(args.keep_nets, args.keep_logs, args.keep_raw)
