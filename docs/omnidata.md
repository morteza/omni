# TODO: OmniData Specification (Draft)

Human data comes in many modalities and flavours. Here, I will try to define a common dataset format that can be used to store and retrieve cognitive data, mainly addressing the needs on the OmniBrain project. 

## Minimal formats

- rs-fMRI is a 3D timeseries with some metadata on sampling, transformations, and acquisition. Indices are XYZ of the voxels in a standard space, and a time dimension. Processed datasets also include structural T1w images, additional binary mask, field-maps, and parcellation (mapping of voxels to regions). Alternatively, processed data can become 2D (region x time).

- rs-EEG is a 2D timeseries of channels x time. Each channel has a location in 3D space. Metadata includes the sampling rate, the montage, and the acquisition parameters. The time dimension is in seconds or milliseconds, and signal is measured in microvolt, v, etc. Common formats are EDF, BrianVision, etc. Processed datasets also include power spectral density (PSD). Preprocessed datasets may also include source-level timeseries; these are time-varying voltages for specific regions (rather than electrodes).

- Connectivity datasets are summary statistics the reflect some sort of correlation between regions, electrodes, networks, etc. They come in two forms: static (averaged over time) and dynamic (time-varying).

- Behavioral datasets are in tabular format and reflect either RL-like interactions (events), or summary statistics (trials, blocks, scores, etc.). They are in two forms: self-reports (questionnaires, surveys, etc.) and cognitive tasks (reaction times, accuracy, etc.).


- Eye tracking

