# LTSpBatchSim

The goal of the tool is to allow parameterized simulation runs on a spice circuit. It creates graphs, and uses transient analysis (```.TRAN```) or AC analysis (```.AC```) of LTSpice.

It is tested on MacOS, and should work identically under linux. It is not tested under windows, but it would be easy to adapt.

It is capable of creating one or more graphs per job (all in 1 png file), and multiple jobs per config file. Example:

![simple](img/simple.png "Simple graph")

See more examples below.

Only 1 signal will be shown per graph, but potentially in multiple lines, each one for a separate simulation.
Multiple output signals can be shown, but they will be each in their own graphs, one row per signal.
Each job is either a Transient analysis, either an AC analysis.
For transient analysis, different time scales or zoomed sections can be shown, each in their own column.
For AC analysis, gain and phase each have their own column, or be merged in the same column.

## Requirements for installation

* python3
* pip packages PyLTSpice, matplotlib (as usual: ```pip install -i requirements.txt```)
* WINE (the MacOS version of LTSPice is incomplete with regards to command line options, linux even has no choice)
* ltspice installed (and updated and run at least once manually) under wine

## Requirements for run

* a circuit, be it in .asc, .cir or .net format
* a json file describing the simulations to run

## How to run

Example:
```python3 ltspbatchsim.py opamptest.json```

See the output of ```python3 ltspbatchsim.py -h``` for more info.

```text
usage: ltspbatchsim.py [-h] [--ltspicepath LTSPICEPATH] [--outdir OUTDIR] [--use_asc] [--keep_nets] [--keep_logs] [--keep_raw] [--single_bode]
                       [--dense]
                       config_file [job_name ...]

Runs one or more LTSpice simulations based on config from a json file. Will use LTSpice installed under wine.

positional arguments:
  config_file           Name of the config json file. Default: 'ltspbatchsim.json'
  job_name              Name of the job(s). If left empty: all jobs will be run.

options:
  -h, --help            show this help message and exit
  --ltspicepath LTSPICEPATH
                        Path of ltspice. Default: '/Users/me/.wine/drive_c/Users/me/AppData/Local/Programs/ADI/LTspice/LTspice.exe'
  --outdir OUTDIR       Output directory for the graphs, also work directory. Default: './batchsim/'
  --use_asc             Run the simulations as usual, but do that using .asc files. This is somewhat slower, and has various issues (see
                        spicelib issues on github), but can be useful for diving into problems detected with the simulations, as it keeps the
                        .asc files after the simulations.
  --keep_nets           After the runs, keep the netlists.
  --keep_logs           After the runs, keep the spice run logs.
  --keep_raw            After the runs, keep the .raw files.
  --single_bode         Keep AC analysis bode plots in the same graph, instead of having gain and phase in separate columns.
  --dense               Use this if the graph is dense. It will dash the lines, making distinction easier. Not used with '--single_bode'
```

# the JSON config file format

```text
model: str
ylabels: [str, ...]
ac: str
transients: [str, ...]
defs: dict(dict)
alt: int|bool
run
 |-name: str
 |-op: str
 |-alt: int|bool
 |-ylabel: str
 |-ylabels: [str,...]
 |-ac: str
 |-transient: str
 |-transients: [str,...]
 |-traces: [str,...]
 |-commondefs: [dict|str,...]
 |-tracedefs: [dict|str,...]
```

* ```model```: the file name of the circuit.
* ```ylabels```: the signals to be shown Each signal will get its own row. These are the default signals for all jobs, and can be overriden in the jobs.
* ```ac```: the default values for the AC analysis of all AC analysis jobs. Can be overriden in the jobs. Format is identical to the spice ```.ac``` op command. Ignored when Transient analysis is requested by the job.

    Exemple: ```"dec 200 5 10e6"```

* ```transients```: the default values for the time sections of all transient analysis jobs. Can be overriden in the jobs. Ignored when AC analysis is requested by the job.
  
    Format: ```[str, ...]```, where each ```str```, inspired by the spice ```.tran``` op command: "Tstop" or "Tstop Tstart" or "0 Tstop Tstart" (=> Tstep must be 0 if specified, dTmax is not used). Additionally, a 'd' can be added at the end designating that this a dense graph, see '--dense'.
  
    Example: ```["10u", "1010u 1000u", "2011u 2001u", "3011u 3001u", "4m d"]``` , creating a large graph with the following sub-graphs in columns:
  * 10 µsecs wide, starting at T0
  * 10 µsecs wide, starting at 1 msecs
  * 10 µsecs wide, starting at 2.001 msecs
  * 10 µsecs wide, starting at 3.001 msecs
  * 4 msecs wide, starting at T0, with dashed lines (to better differentiate them)
* ```defs```: a set of common component value definitions, to be referred to by the jobs and the traces

    Example:

```json
    "defs": {
        "HighV": {
            "R22": "300k",
            "V3": "100"
        },
        "LowV": {
            "R22": "20k",
            "V3": "10"
        },
        "Load_ON": { "V6": "1" },
        "Load_OFF": { "V6": "0" },
        "Load_Pulse": {
            "V6": "PWL(1u 0 +2n 1 +1m 1 +2n 0 +1u 0 +2n 1 +1m 1 +2n 0 +1m 0 +2n 1 +1u 1 +2n 0 +1m 0)"
        }
    }
```

* ```alt```: (true|false|1|0) Determine the use of the normal solver or the alternate solver. This is the default value. Can be overriden in the jobs.
* ```run```: the definition of the jobs
* ```run.name```: the name of the job
* ```run.op```: the type of analysis. Set to 'ac[N]' for AC analysis.
  * 'ac' for AC analysis bode plot creation according to the ```--single_bode``` command line setting
  * 'ac1' for AC analysis bode plot in a single graph
  * 'ac2' for a bode plot in separate gain and phase graphs
  * If absent, or any other value, a transient analysis is performed. 
  * When using 'ac', be sure to designate a signal source, and to define the signal level. Example: ```"V3": "0 AC 1"```
* ```run.alt```: Will override root level ```alt``` mentioned above.
* ```run.ylabel```: Will override root level ```ylabels``` mentioned above. Used when only 1 value is needed.
* ```run.ylabels```: Will override root level ```ylabels``` mentioned above.
* ```run.ac```: Will override root level ```ac``` mentioned above.
* ```run.transient```: Will override root level ```transients``` mentioned above. Used when only 1 value is needed.
* ```run.transients```: Will override root level ```transients``` mentioned above.
* ```run.traces```: List of the individual traces inside a graph. These names are not only printed, but also used to set component values. See ```run.tracedefs```.
* ```run.commondefs```: component value settings for all traces in the graph. Can be a ```str```, in which case it refers to a deinition in the ```defs``` section at the root of the json file. If it is a ```dict```, it is a key/value list of component names and values.
* ```run.tracedefs```: templatized component value settings for all traces in the graph. Is like ```commondefs```, but also accepts variables that are derived from ```traces```. If ```{name}``` is specified in a value in ```tracedefs```, it will be substituted by the name of the trace. If ```{nameN}``` is specified in a value in ```tracedefs```, it will be substituted by the Nth part of the name of the trace, split by comma or space.

    Examples:

```json
    {
    "defs": {
        "sense_1uA": { "RIsense": "5MEG" },
        "sense_10uA": { "RIsense": "560k" },
        "sense_100uA": { "RIsense": "50k" },
        "sense_1mA": { "RIsense": "5k" },
        "sense_10mA": { "RIsense": "500" },
        "sense_100mA": { "RIsense": "50" },
    },
    ...
    "run": [
        {
            ...
            "tracedefs": ["sense_{name}"],
            "traces": ["1uA", "10uA", "100uA", "1mA", "10mA", "100mA"]
    ...
```

```json
  "run": [
    {
      ...
      "tracedefs": [
        {
          "XU6": "{name1}",
          "C3": "{name2}",
          "C4": "{name2}",
          "C21": "{name3}",
          "C22": "{name3}"
        }
      ],
      "traces": [
        "OPAx145, 27pf, 0pf",
    ...
```

# About use_asc

This has various issues. I have detected problems with AC analysis and vertical directives. See https://github.com/nunobrum/spicelib/issues.


# Examples of graphs

(The view below uses adaptive scaling, download to see the full resolution)

![more complicated](img/more.png "More detailed graph")

![even more complicated](img/moremore.png "Even more detailed graph")

![bode 2](img/bode2.png "Bode plot in 2 graphs")

![bode 1](img/bode1.png "Bode plot in 1 graph")

# Hints

You can use MacOS's LTSpice to draw and this tool (thus wine) to do the batched simulations. Just don't close MacOS's LTSpice _while_ doing a batch run, as it (by default) will delete the (temporary) .net file that got created by this tool.
