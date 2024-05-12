# LTSpBatchSim

The goal of the tool is to allow parameterized simulation runs on a spice circuit. It creates graphs, and uses transient analysis (```.TRAN```) of LTSpice.

It is tested on MacOS, and should work identically under linux. It is not tested under windows, but it would be easy to adapt.

It is capable of creating one or more graphs per job (all in 1 png file), and multiple jobs per config file. Example:

![simple](img/simple.png "Simple graph")

See more examples below.

Only 1 signal will be shown per graph, but potentially in multiple lines, each one for a separate simulation.
Multiple output signals can be shown, but they will be each in their own graphs, one row per signal.
Different time scales or zoomed sections can be shown, each in their own column.

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
transients: [str, ...]
defs: dict(dict)
run
 |-name: str
 |-ylabel: str
 |-ylabels: [str,...]
 |-transient: str
 |-transients: [str,...]
 |-traces: [str,...]
 |-commondefs: [dict|str,...]
 |-tracedefs: [dict|str,...]
```

* ```model```: the file name of the circuit.
* ```ylabels```: the signals to be shown Each signal will get its own row. These are the default signals for all jobs, and can be overriden in the jobs.
* ```transients```: the values for the time sections. These are the default time sections for all jobs, and can be overriden in the jobs. 
  
    Format: ```[str, ...]```, where each ```str```, inspired by the spice ```.tran``` op command: "Tstop" or "Tstop Tstart" or "0 Tstop Tstart" (=> Tstep must be 0 if specified, dTmax is not used). Only integer times are allowed, expressed in µsecs, msecs or full seconds.
  
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
* ```run.ylabel```: Will override ```ylabels``` above. Used when only 1 value is needed.
* ```run.ylabels```: Will override ```ylabels``` above.
* ```run.transient```: Will override ```transients``` above. Used when only 1 value is needed.
* ```run.transients```: Will override ```transients``` above.
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

# Hints

You can use MacOS's LTSpice to draw and this tool (thus wine) to do the batched simulations. Just don't close MacOS's LTSpice _while_ doing a batch run, as it (by default) will delete the (temporary) .net file that got created by this tool.
