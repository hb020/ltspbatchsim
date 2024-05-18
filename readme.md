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

See the output of ```python3 batchsim.py -h``` for more info.

# the JSON config file format

```text
model: str
ylabels: [str, ...]
ac: str
transients: [str, ...]
defs: dict(dict)
run
 |-name: str
 |-op: str
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
  
    Format: ```[str, ...]```, where each ```str```, inspired by the spice ```.tran``` op command: "Tstop" or "Tstop Tstart" or "0 Tstop Tstart" (=> Tstep must be 0 if specified, dTmax is not used). Times shorter than µsecs are not supported.
  
    Example: ```["10u", "1010u 1000u", "2011u 2001u", "3011u 3001u", "4m"]``` , creating a large graph with the following sub-graphs in columns:
  * 10 µsecs wide, starting at T0
  * 10 µsecs wide, starting at 1 msecs
  * 10 µsecs wide, starting at 2.001 msecs
  * 10 µsecs wide, starting at 3.001 msecs
  * 4 msecs wide, starting at T0
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

* ```run```: the definition of the jobs
* ```run.name```: the name of the job
* ```run.op```: the type of analysis. Set to 'ac' for AC analysis. If absent, or any other value, a transient analysis is performed. When using 'ac', be sure to designate a signal source, and to define the signal level. Example: ```"V3": "0 AC 1"```
* ```run.ylabel```: Will override root level ```ylabels``` mentioned above. Used when only 1 value is needed.
* ```run.ylabels```: Will override root level ```ylabels``` mentioned above.
* ```run.ac```: Will override root level ```ac``` mentioned above.
* ```run.transient```: Will override root level ```transients``` mentioned above. Used when only 1 value is needed.
* ```run.transients```: Will override root level ```transients``` mentioned above.
* ```run.traces```: List of the individual traces inside a graph. These names are not only printed, but also used to set component values. See ```run.tracedefs```.
* ```run.commondefs```: component value settings for all traces in the graph. Can be a ```str```, in which case it refers to a deinition in the ```defs``` section at the root of the json file. If it is a ```dict```, it is a key/value list of component names and values.
* ```run.tracedefs```: templatized component value settings for all traces in the graph. Is like ```commondefs```, but accepts variables that are derived from ```traces```. If ```{name}``` is specified in a value in ```tracedefs```, it will be substituted by the name of the trace. If ```{nameN}``` is specified in a value in ```tracedefs```, it will be substituted by the Nth part of the name of the trace, split by comma or space.

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

# Examples of graphs

![more complicated](img/more.png "More detailed graph")

![even more complicated](img/moremore.png "Even more detailed graph")

![bode](img/bode1.png "Bode plot")

# Hints

You can use MacOS's LTSpice to draw and this tool (thus wine) to do the batched simulations. Just don't close MacOS's LTSpice _while_ doing a batch run, as it (by default) will delete the (temporary) .net file that got created by this tool.
