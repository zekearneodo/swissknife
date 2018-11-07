# swissknife
Multiple tools to:

- post-process neural data recorded in singing/anesthetized with the Intan RHD2000 system using both Intan and OpenEphys software.
- convert it to the Klusta-team's Kwik format.
- sort spikes using klusta/kilosort/mountainsort, or simply detect supra-threshold events.
- align the neural data with the stimuli played/bouts detected
- fit parameters and create synthetic vocalizations by integrating the Mindlin model for the vocal organ of the zebra finch.
- Use a FFNN/LSTM/Linear decoder to produce synthetic vocalizations driven by neural activity.

(Documentation and splitting into more reasonably sized, stand-alone packages is in progress).
