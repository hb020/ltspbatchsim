import time
import numpy as np
import spicelib
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.sim.simulator import Simulator
from matplotlib import pyplot as plt
from matplotlib import axes
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
from pathlib import Path

# Run a series of simulations on a model, with varying variable values. Puts the result in one single png (per job).

outdir = "./batchsim/"

# default config file
default_config_file = "ltspbatchsim.json"

# global object with the configuration read from the config file
CONFIG = {}

# the additional files to be included in the simulation
additional_files = ["*.lib", "*.LIB", "*.asy", "*.ASY"]

# global array for the temporary filenames (actually, the base names)
tmp_filenames = []


def delfiles(pattern: str):
    """Delete files based on a pattern.

    :param pattern: file name pattern, with wildcards
    :type pattern: str
    """
    for p in glob.glob(pattern, recursive=False):
        if os.path.isfile(p):
            os.remove(p)


def tmp_filenames_register(fname: str):
    """Register a temporary file name for later deletion.

    :param fname: base file name, no extension
    :type fname: str
    """
    global tmp_filenames
    tmp_filenames.append(fname)
    

def tmp_filenames_cleanup(keep_asc: bool = False, keep_net: bool = False, keep_log: bool = False, keep_raw: bool = False):
    """Clean up the temporary files, allow some files to be kept

    :param keep_asc: Keep ".asc" files, defaults to False
    :type keep_asc: bool, optional
    :param keep_net: Keep ".net" files, defaults to False
    :type keep_net: bool, optional
    :param keep_log: Keep ".log" files, defaults to False
    :type keep_log: bool, optional
    :param keep_raw: Keep ".raw" files, defaults to False
    :type keep_raw: bool, optional
    """
    for fname in tmp_filenames:
        if not keep_asc:
            delfiles(f"{fname}*.asc")            
        if not keep_net:
            delfiles(f"{fname}*.net")
        if not keep_log:
            delfiles(f"{fname}*.log")
        if not keep_raw:
            delfiles(f"{fname}*.raw")
    

def is_ltspice(simulator: Simulator) -> bool:
    return "ltspice" in simulator.__name__.lower()
    
# format the graph axis
def format_axis_transient(ax: axes.Axes, size: int, offset_time_s: float, duration_time_s: float,
                          ylabel: str, jobname: str):
    """Format the axis for a transient analysis.

    :param ax: the axis to format
    :type ax: axes.Axes
    :param size: size of the sections, 0 for automatic sectionning
    :type size: int
    :param offset_time_s: offset time in seconds
    :type offset_time_s: float
    :param duration_time_s: duration time in seconds
    :type duration_time_s: float
    :param ylabel: the label for the y axis
    :type ylabel: str
    :param jobname: the job name
    :type jobname: str
    """
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


def format_axis_ac(ax: axes.Axes, is_gain: bool, single_bode: bool, x_scale: str, 
                   start_freq: float, end_freq: float, ylabel: str, jobname: str):
    """Format the axis for an AC analysis.

    :param ax: The axis to format
    :type ax: axes.Axes
    :param is_gain: if this is a gain plot
    :type is_gain: bool
    :param single_bode: if this is a single bode plot
    :type single_bode: bool
    :param x_scale: the x scale, can be "lin" or "log"
    :type x_scale: str
    :param start_freq: the start frequency, in Hz
    :type start_freq: float
    :param end_freq: the end frequency, in Hz
    :type end_freq: float
    :param ylabel: the label for the y axis
    :type ylabel: str
    :param jobname: the job name
    :type jobname: str
    """
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


def str2float(s: str) -> float:
    """Transform a string to a float. The string can have a suffix, like "k" for kilo, "m" for milli, etc.

    :param s: input string
    :type s: str
    :return: the resulting float, or math.nan if the string is not a number
    :rtype: float
    """
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
def time2nsecs(t: str) -> int:
    """Transform a time string to nano seconds.

    :param t: input time string, can have a suffix like "n" for nano, "u" for micro, etc.
    :type t: str
    :return: nano seconds or -1 if the string is not a time
    :rtype: int
    """
    f = str2float(t)
    if math.isnan(f):
        return -1
    return round(1000000000.0 * f)


def freq2Hz(t: str) -> float:
    """Transform a frequency string to Hz.

    :param t: frequency string, can have a suffix like "k" for kilo, "m" for milli, etc.
    :type t: str
    :return: herz or math.nan if the string is not a frequency
    :rtype: float
    """
    f = str2float(t)
    return f


def run_job(job: object, jobnr: int, nrjobs: int, take_me: bool = True, 
            model_fname: str = "", simulator: Simulator = LTspice, defaultac: str = "", 
            defaulttransients: list = [], defaultlabels: list = [], 
            single_bode: bool = False, use_asc: bool = False, dense: bool = False, 
            defaultaltsolver: bool = False, timeout: int = None, dryrun: bool = False) -> bool:
    """Run a simulation job, and create a graph from the results.

    :param job: Job object, from config file. The name of the job is to be set as the "name" element
    :type job: object
    :param jobnr: Job nr of the total list of jobs
    :type jobnr: int
    :param nrjobs: Total list of jobs
    :type nrjobs: int
    :param take_me: Run me or not, defaults to True
    :type take_me: bool, optional
    :param model_fname: File name of the model, defaults to ""
    :type model_fname: str, optional
    :param simulator: The simulator to be used, defaults to LTspice
    :type simulator: Simulator, optional
    :param defaultac: Values for the AC analysis. Can be overriden in the job. Format is identical to the spice ```.ac``` op command. \
        Ignored when Transient analysis is requested by the job. Defaults to ""
    :type defaultac: str, optional
    :param defaulttransients: Values for the time sections of all transient analysis jobs. Can be overriden in the job. \
        Ignored when AC analysis is requested by the job. Defaults to []
    :type defaulttransients: list, optional
    :param defaultlabels: The signals to be shown. Each signal will get its own row. These are the default signals for all jobs, \
        and can be overriden in the job. Defaults to []
    :type defaultlabels: list, optional
    :param single_bode: Put gain and phase in the same graph. Defaults to False
    :type single_bode: bool, optional
    :param use_asc: Use the ASC file, not a netlist. Defaults to False
    :type use_asc: bool, optional
    :param dense: Pack the graph densely. Defaults to False
    :type dense: bool, optional
    :param defaultaltsolver: Use the Alt solver. Defaults to False
    :type defaultaltsolver: bool, optional
    :param timeout: Run timeout in seconds. Defaults to None
    :type timeout: int, optional
    :param dryrun: Do not run the simulations, just generate the simulation input files. Defaults to False
    :type dryrun: bool, optional
    :return: True if succeeded. If not, it will log the error.
    :rtype: bool    
    """
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
        return True
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
            return False
        
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
            return False
        
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
        minstep_nsecs = 0  # "undef"
        # transient time format is a subset of the spice format for .TRAN: 
        # "<Tstop>"
        # "<Tstep> <Tstop>"
        # "<Tstep> <Tstop> <Tstart>" (=> dTmax is not used)
        for t in transients:
            step_nsecs = -1
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
                step_nsecs = -1
                endrecording_nsecs = time2nsecs(tparts[0])
                startrecording_nsecs = 0
                visible_nsecs = endrecording_nsecs
            elif len(tparts) == 2:
                step_nsecs = time2nsecs(tparts[0])
                endrecording_nsecs = time2nsecs(tparts[1])
                startrecording_nsecs = 0
                visible_nsecs = endrecording_nsecs
            elif len(tparts) == 3:
                step_nsecs = time2nsecs(tparts[0])
                endrecording_nsecs = time2nsecs(tparts[1])                
                startrecording_nsecs = time2nsecs(tparts[2])
                visible_nsecs = endrecording_nsecs - startrecording_nsecs
            else:
                startrecording_nsecs = -1  # signal error
            
            if startrecording_nsecs < 0 or endrecording_nsecs < 0:
                logger.error(f"Unsupported transient time format \"{t}\".")
                logger.error("  Format must be either \"Tstop\" or \"Tstep Tstop\" or \"Tstep Tstop Tstart\". (Like LTSpice .TRAN)")
                return False

            if visible_nsecs <= 0:
                logger.error(f"Unsupported transient time format \"{t}\", duration time cannot be 0 or negative.")
                logger.error("  Format must be either \"Tstop\" or \"Tstep Tstop\" or \"Tstep Tstop Tstart\". (Like LTSpice .TRAN)")
                return False
            
            column_definitions.append({"name": t,
                                       "startrecording_nsecs": startrecording_nsecs, 
                                       "endrecording_nsecs": endrecording_nsecs, 
                                       "visible_nsecs": visible_nsecs,
                                       "dense": t_dense})
            if maxtime_nsecs < endrecording_nsecs:
                maxtime_nsecs = endrecording_nsecs
                
            # tstep is specified?
            if step_nsecs > 0:
                # tstep is specified here
                if minstep_nsecs <= 0 or minstep_nsecs > step_nsecs:
                    # no minstep yet or a smaller tstep than given previously?
                    minstep_nsecs = step_nsecs
        
        # read all times.
        # add some more to the max time, sometimes I get stange effects at the end
        maxtime_nsecs = int(maxtime_nsecs * 1.1)
        # do not use 0 as minstep on NGspice, as it will freak out the simulator
        if minstep_nsecs < 1:
            if is_ltspice(simulator):  # LTspice can handle 0
                minstep_nsecs = 0
            else:
                minstep_nsecs = (maxtime_nsecs / 1000)
                if minstep_nsecs < 1:
                    minstep_nsecs = 1
                
    nrcols = len(column_definitions)
    # print(column_definitions)
                 
    if nrcols == 0 or nrrows == 0:
        logger.error(f"No plots defined for {jobname}")
        return False

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
        if is_ltspice(simulator):
            encoding = "autodetect"  # LTSpice can use a lot of different encodings, even on the same platform
        else:
            encoding = "utf-8"
        netlist = spicelib.SpiceEditor(f"./{model_fname}", encoding=encoding)  # Open the Spice Model, and creates the tailored .net file

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
            netlist.add_instruction(f".tran {minstep_nsecs}n {maxtime_nsecs}n")        
        
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
        spicelogfile = f"{basename}_{safe_tracename}.log"
        processlogfile = f"{basename}_{safe_tracename}_process.log"
        
        s = "defs"
        if s in job: 
            for k in job[s]:
                d = {}
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
                        
                for kk, kv in d.items():
                    if kk.startswith("X"):
                        if use_asc:
                            # remove the X when using spicelib.AscEditor
                            kk = kk[1:]
                    logger.debug(f"set_component_value({kk}:{kv})")
                    # logger.debug(f" before set: get_component_value({kk}) = {netlist.get_component_value(kk)}")
                    # logger.debug(f" before set: get_component_parameters({kk}) = {netlist.get_component_parameters(kk)}")
                    netlist.set_component_value(kk, kv)
                    # logger.debug(f" after set: get_component_value({kk}) = {netlist.get_component_value(kk)}")
                    if use_asc:
                        # wipe Value2 if in asc
                        netlist.set_component_parameters(kk, **{"Value2": ""})                            
                                            
        netlist.save_netlist(netlistfile)
        
        logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Trace {traceidx}/{nrtraces} '{tracename}'")
        
        if not dryrun:
            opts = []
            if is_ltspice(simulator):
                if alt_solver:
                    opts.append('-alt')
                else:
                    opts.append('-norm')
                
            with open(processlogfile, "w") as outfile:
                # keep timing, so I can print the process time
                starttm = time.time()
                try:
                    rv = simulator.run(netlistfile, opts, timeout=timeout, stdout=outfile, stderr=subprocess.STDOUT)
                except subprocess.TimeoutExpired:
                    logger.error(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Trace {traceidx}/{nrtraces} '{tracename}' timed out, exiting.")
                    return False
                endtm = time.time()
                logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Trace {traceidx}/{nrtraces} '{tracename}' took {endtm - starttm:.1f} seconds, result = {rv}")
            
            # interpret the simulation results
            if not os.path.exists(rawfile):
                logger.error(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Trace {traceidx}/{nrtraces} '{tracename}' failed. The last line of the log file \"{spicelogfile}\":")                
                # default log:
                last_line = f"(No log file \"{spicelogfile}\" found.)"
                if os.path.exists(spicelogfile):
                    with open(spicelogfile, "rt") as f:
                        last_lines = f.readlines()
                        # use one of the 2 last lines, if they are not empty
                        if len(last_lines) >= 2:
                            last_line1 = last_lines[-1].strip()
                            last_line2 = last_lines[-2].strip()
                            if len(last_line1) > 1:
                                last_line = last_line1
                            elif len(last_line2) > 1:
                                last_line = last_line2
                logger.error(last_line)
                return False
            
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

    if not dryrun:
        logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", Creating graph.")
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
        logger.info(f"Job {jobnr}/{nrjobs}: \"{jobname}\", results are in \"{imagefile}\"")
    return True
    

def prepare():
    """Prepare the output directory and copy the additional files.
    """
    for cf in additional_files:
        for f in glob.glob(cf):
            shutil.copyfile(f"./{f}", f"{outdir}{f}")


def load_config(config_file: str):
    """Load the configuration from a json file.

    :param config_file: file name of the json file
    :type config_file: str
    """
    global CONFIG
    logger.info(f"Using config file \"{config_file}\".")
    f = open(config_file)
    CONFIG = json.load(f)


def show_paths_of_simulator(simulator: Simulator, name: str):
    """Show the paths of the simulator.

    :param simulator: The simulator to show the paths of
    :type simulator: Simulator
    :param name: The name of the simulator
    :type name: str
    """
    print(f"Simulator: {name}")    
    if not simulator.is_available():
        print("  Not found.")
        return
    if len(simulator.spice_exe) == 1:
        winepath = None
        spice_path = simulator.spice_exe[0]
    else:
        winepath = simulator.spice_exe[0]
        spice_path = simulator.spice_exe[1]
    if winepath:
        print(f"  Wine path: '{winepath}'")
    print(f"  Spice path: '{spice_path}'")
    print(f"  Spice process name: '{simulator.process_name}'")
    print(f"  Default library paths: {simulator.get_default_library_paths()}")


def show_paths():
    show_paths_of_simulator(LTspice, 'LTspice')
    show_paths_of_simulator(NGspiceSimulator, 'NGSpice')
    

if __name__ == "__main__":
    """Main program.
    """
    # default values
    DEFAULT_SPICE = "ltspice"
    SPICE_CHOICES = ['ltspice', 'ngspice']
    
    simulator = None       
       
    parser = argparse.ArgumentParser(description="Runs one or more LTSpice or NGSpice simulations based on config from a json file.")
    parser.add_argument('config_file', default=default_config_file, help=f"Name of the config json file. Default: '{default_config_file}'.") 
    parser.add_argument('job_name', default="", nargs="*", help="Name of the job(s) to run. If left empty: all jobs will be run. Wildcards can be used, "
                        "but please escape the * and ? to avoid shell expansion. Example of good use in shell: \"test_OPA189\\*\", "
                        "which will be passed on to this program as \"test_OPA189*\".") 
    parser.add_argument('-v', '--verbose', default='', help="Be verbose", action="store_const", dest="loglevel", const='v')
    parser.add_argument('-vv', '--debug', default='', help="Be more verbose", action="store_const", dest="loglevel", const='vv')
    parser.add_argument('--log', default=False, action='store_true', help="Log to file, not to console. "
                        "If set, will log to \"{config_file}.log\", in append mode.")
    parser.add_argument('--sim', type=str.lower, default=DEFAULT_SPICE, choices=SPICE_CHOICES,
                        help=f"Simulator to be used, default: '{DEFAULT_SPICE}'.")
    parser.add_argument('--spicepath', default=None, help="Path of the spice executable.")
    if sys.platform == 'linux' or sys.platform == 'darwin':
        parser.add_argument('--winepath', default=None, help="Path of the wine executable, if used.")
    parser.add_argument('--showpaths', default=False, action='store_true', help="Show the executable and library paths that can be detected automatically.")
    parser.add_argument('--outdir', default=outdir, help=f"Output directory for the graphs, also work directory. Default: '{outdir}'.")
    parser.add_argument('--dryrun', default=False, action='store_true', help="Do not run the simulations, just generate the simulation input files. "
                        "This implies --keep_simfiles and can be used with --use_asc.")
    parser.add_argument('--use_asc', default=False, action='store_true', help="Run the simulations directly from .asc files, not from .net files. "
                        "There may be some bugs, so use with caution. This only works with LTSpice.")
    parser.add_argument('--keep_simfiles', default=False, action='store_true', help="After the runs, keep the simulation input files, "
                        "be it .net or .asc (when used with --use_asc).")
    parser.add_argument('--keep_logs', default=False, action='store_true', help="After the runs, keep the spice run logs.")
    parser.add_argument('--keep_raw', default=False, action='store_true', help="After the runs, keep the .raw files.")
    parser.add_argument('--single_bode', default=False, action='store_true', help="Keep AC analysis bode plots in the same graph, "
                        "instead of having gain and phase in separate columns.")
    parser.add_argument('--dense', default=False, action='store_true', help="Use this if the graph is dense. "
                        "It will dash the lines, making distinction easier. Not used with '--single_bode'.")
    
    # use_asc is not preferred, as spicelib has some limitations in the spicelib.AscEditor.
    
    args = parser.parse_args()
    
    if args.showpaths:
        show_paths()
        exit(0)
        
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

    # default log levels
    lib_loglevel = logging.WARNING
    my_loglevel = logging.INFO
    
    # now set log level according to the arguments
    if args.loglevel == 'v':
        my_loglevel = logging.DEBUG
        lib_loglevel = logging.INFO
    elif args.loglevel == 'vv':
        my_loglevel = logging.DEBUG
        lib_loglevel = logging.DEBUG
        
    if args.log:
        fname, fext = os.path.splitext(args.config_file) 
        logfile = os.path.join(outdir, fname + ".log")
        print(f"logging to \"{logfile}\"")
        # configure logging
        logging.basicConfig(filename=logfile, level=my_loglevel)
    else:
        logging.basicConfig(level=my_loglevel)
        
    logger = logging.getLogger(Path(__file__).stem)
    
    spicelib.set_log_level(lib_loglevel)
                        
    spicepath = args.spicepath
    if sys.platform == 'linux' or sys.platform == 'darwin':
        winepath = args.winepath
        if winepath and len(winepath) == 0:
            winepath = None
    else:
        winepath = None
        
    if spicepath and not (os.path.isfile(spicepath) or shutil.which(spicepath)):
        logger.error(f"spice is not found under \"{spicepath}\". Please specify a valid path via --spicepath.")
        exit(1)
        
    if winepath and not (os.path.isfile(winepath) or shutil.which(winepath)):
        logger.error(f"wine is not found under \"{winepath}\". Please specify a valid path via --winepath.")
        exit(1)            

    # determine the simulator
    if args.sim == "ltspice":
        simulator = LTspice        
    if args.sim == "ngspice":
        simulator = NGspiceSimulator
    logger.info(f"Using simulator: {args.sim}")
    # logger.info(f"Simulator class == LTspice: {is_ltspice(simulator)}")  # this works
    # logger.info(f"Simulator inst == LTspice: {isinstance(simulator, LTspice)}")  This fails, as it is a class
    # logger.info(f"Simulator class name: {simulator.__name__}")

    # set the path for the spice executable
    if winepath or spicepath:
        old_exe = simulator.spice_exe
        new_winepath = None
        new_spicepath = None
        
        if len(old_exe) == 1:
            new_winepath = None
            new_spicepath = old_exe[0]
        if len(old_exe) == 2:
            new_winepath = old_exe[0]
            new_spicepath = old_exe[1]
            
        if winepath:
            new_winepath = winepath
        if spicepath:
            new_spicepath = spicepath
            
        if new_winepath:
            simulator.spice_exe = [new_winepath, new_spicepath]
        else:
            simulator.spice_exe = [new_spicepath]
        simulator.process_name = simulator.guess_process_name(simulator.spice_exe[0])  # let the lib find out the process name

    if not simulator.is_available():
        logger.error(f"Cannot find the simulator at {simulator.spice_exe}.")
        exit(1)
    
    # set the paths for the libraries.
    if is_ltspice(simulator):
        spicelib.AscEditor.prepare_for_simulator(simulator)
    spicelib.SpiceEditor.prepare_for_simulator(simulator)
                    
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
        if not is_ltspice(simulator):
            logger.error("This simulator does not handle .asc files. Use LTspice.")
            exit(1)            
    else:
        if model_ext.lower() == ".asc":
            if not is_ltspice(simulator):
                logger.error("This simulator does not handle .asc files. Use LTspice for those files.")
                exit(1)            
            # convert to .net file
            logger.info(f"Creating the netlist for the schema \"{model_fname}\"")
            # make sure the log file goes to outdir
            processlogfile = os.path.join(outdir, f"{model_root}_create_netlist.log")
            tmp_filenames_register(processlogfile[:-4])  # add the log file to the list of files to clean up
            with open(processlogfile, "w") as outfile:
                model_fname = simulator.create_netlist(model_fname, stdout=outfile, stderr=subprocess.STDOUT)

            if not os.path.isfile(model_fname):
                logger.error(f"Cannot find netlist file \"{model_fname}\", something went wrong. See file {processlogfile}.")
                exit(1)

    logger.info(f"Running simulations, on netlist \"{model_fname}\".")
    if len(my_jobs) != 0:
        jobnames = ', '.join(f'\"{w}\"' for w in my_jobs)
        logger.info(f"Limiting to jobs with names {jobnames}.")        
        
    if "description" in CONFIG:
        description = CONFIG["description"]
        logger.info(f"Description: {description}")
    
    timeout = None
    if "timeout" in CONFIG:
        timeout = int(CONFIG["timeout"])
            
    nrjobs = len(CONFIG["jobs"])
    jobnr = 0
    for jobname, job in CONFIG["jobs"].items():
        jobnr += 1
        job["name"] = jobname
        # look if I need to run this job
        take_me = True
        if len(my_jobs) > 0:
            take_me = False
            for j in my_jobs:
                j = j.lower()
                if '*' in j or '?' in j:
                    if fnmatch.fnmatch(job["name"].lower(), j):
                        take_me = True
                        break                        
                else:
                    if job["name"].lower() == j:
                        take_me = True
                        break
                
        if not run_job(job, jobnr, nrjobs, 
                       take_me, 
                       simulator=simulator,
                       model_fname=model_fname,
                       defaultac=defaultac, 
                       defaulttransients=defaulttransients, 
                       defaultlabels=defaultlabels,
                       single_bode=args.single_bode,
                       use_asc=args.use_asc,
                       dense=args.dense,
                       defaultaltsolver=defaultaltsolver,
                       timeout=timeout,
                       dryrun=args.dryrun
                       ): 
            logger.error("Error: bailing out.")
            exit(1)
    
    # cleanup afterwards
    keep_asc = False
    keep_net = False
    if args.use_asc:
        keep_asc = args.dryrun or args.keep_simfiles
    else:
        keep_net = args.dryrun or args.keep_simfiles
        
    tmp_filenames_cleanup(keep_asc=keep_asc, keep_net=keep_net, keep_log=args.keep_logs, keep_raw=args.keep_raw)
