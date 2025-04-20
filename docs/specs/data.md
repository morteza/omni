# TODO: OmniData Specification (Draft)

Human data comes in many modalities and flavours. Here, I will try to define a common dataset format that can be used to store and retrieve cognitive data, mainly addressing the needs on the Omni-Brain project. 

## Minimal formats

- rs-fMRI is a 3D timeseries with some metadata on sampling, transformations, and acquisition. Indices are XYZ of the voxels in a standard space, and a time dimension. Processed datasets also include structural T1w images, additional binary mask, field-maps, and parcellation (mapping of voxels to regions). Alternatively, processed data can become 2D (region x time). Shapes:
    - 5D: subject X Y Z time
    - 3D: subject region time
    - Slices??

- rs-EEG is a 2D timeseries of channels x time. Each channel has a location in 3D space. Metadata includes the sampling rate, the montage, and the acquisition parameters. The time dimension is in seconds or milliseconds, and signal is measured in microvolt, v, etc. Common formats are EDF, BrianVision, etc. Processed datasets also include power spectral density (PSD). Preprocessed datasets may also include source-level timeseries; these are time-varying voltages for specific regions (rather than electrodes). Shapes:
    - 3D: subject channel time
    - 3D: subject region time
    - segments: subject region patch time

- Connectivity datasets are summary statistics the reflect some sort of correlation between regions, electrodes, networks, etc. They come in two forms: static (averaged over time) and dynamic (time-varying). Shape
    - Static: subject region region
    - Static: subject channel channel
    - Dynamic: subject region region time
    - Dynamic: subject channel channel time

- Behavioral datasets are in tabular format and reflect either RL-like interactions (<s,a,r> from the agents point of view), events (form the system point of view), or summary statistics (scores, etc). Shapes
    - agents: subject episode [step] s a r
    - events (in env): subject time event
    - statistics (e.g., trial-by-trial): subject episode stat
    - statistics (e.g., personality scores): subject stat


- Eye tracking datasets are in tabular format and represents the position of the eye in 3D space (x,y,z) and the time of the sample. The data is usually sampled at specific sampling rate. The data can be used to compute fixations, saccades, etc. Shapes:
    - subject time x y z
    - subject time [x y z] [fixation]
    - subject time [x y z] [saccade]
    - statistics

