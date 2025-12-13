from dataclasses import dataclass
from io import BufferedReader, BytesIO
from pathlib import Path
from struct import unpack
import time
import numpy as np
import os
from math import floor

DEBUG = False

@dataclass(init=False, repr=False)
class EDF():

    subject_id: str
    recording_id: str
    day: int
    month: int
    year: int
    hour: int
    minute: int
    second: int
    data_offset: int
    subtype: str
    data_size: int
    n_records: int
    record_length: float
    nchan: int
    ch_names: list
    transducers: list
    units: list
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    prefiltering: list
    highpass: float | None
    lowpass: float | None
    n_samps: np.ndarray
    calibrate: np.ndarray
    offset: np.ndarray
    fname: Path

    def __init__(self, fname: Path):
        self.fname = fname
        # HACK to please type checkers
        self.f: BufferedReader = open(os.devnull, "rb")
        self.f.close()

    @classmethod
    def open(cls, fname):
        self = cls(fname)
        return self

    def __enter__(self):
        try:
            if self.f is None or self.f.closed:
                self.f = open(self.fname, "rb")
                self.read_header()
            if DEBUG:
                dts = []
                for i in range(1000):
                    t0 = time.time_ns()
                    self.read_header()
                    dt = time.time_ns() - t0
                    dts.append(dt)
                print(f"EDF header read in {np.mean(dts)/1e6:.2f} ms over 1000 runs")
            return self
        except Exception:
            if self.f is not None and not self.f.closed:
                self.f.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.f is not None and not self.f.closed:
            self.f.close()
        return False  # don't suppress exceptions

    def read_header(self):
        h = {}
        import struct

        # fixed header (depends only on EDF spec order)
        fixed = [
            ("magic",        8,  str),
            ("subject_id",   80, str),
            ("recording_id", 80, str),
            ("start_date",   8,  str),
            ("start_time",   8,  str),
            ("data_offset",  8,  int),
            ("reserved",     44, str),
            ("n_records",    8,  int),
            ("record_length",8,  float),
            ("nchan",        4,  int)
        ]

        fixed_fmt = "".join(f"{width}s" for _, width, _ in fixed)  # e.g., '8s80s80s8s8s8s44s8s8s4s'
        raw = self.f.read(struct.calcsize(fixed_fmt))
        values = struct.unpack(fixed_fmt, raw)

        # parse fixed header
        for (name, width, cast_fn), value in zip(fixed, values):
            h[name] = cast_fn(value.strip().decode())

        # parse data_size
        self.subtype = h["reserved"][:5]
        self.data_size = 3 if self.subtype in ('24BIT', 'bdf') else 2

        per_ch = [
            ("ch_names",      16, str),
            ("transducers",   80, str),
            ("units",         8,  str),
            ("physical_min",  8,  float),
            ("physical_max",  8,  float),
            ("digital_min",   8,  float),
            ("digital_max",   8,  float),
            ("filtering",     80, str),
            ("n_samps",       8,  int),
            ("ch_reserved",   32, str)
        ]

        for name, width, cast_fn in per_ch:
            h[name] = [cast_fn(self.f.read(width).strip().decode()) for _ in range(h["nchan"])]

        # convert numeric lists to numpy arrays
        for k in ("physical_min", "physical_max", "digital_min", "digital_max", "n_samps"):
            h[k] = np.asarray(h[k])

        # set attributes
        for k,v in h.items():
            setattr(self, k, v)

        self.calibrate = (self.physical_max - self.physical_min)/(self.digital_max - self.digital_min)
        self.offset = self.physical_min - self.calibrate * self.digital_min
        for ch in range(self.nchan):
            if self.calibrate[ch]<0:
                self.calibrate[ch] = 1
                self.offset[ch] = 0

    def readBlock(self, block):
        assert(block>=0)
        data = []
        f = self.f

        blocksize = np.sum(self.n_samps) * self.data_size
        f.seek(self.data_offset + block * blocksize)
        for i in range(self.nchan):
            buf = f.read(self.n_samps[i]*self.data_size)
            raw = np.asarray(unpack('<{}h'.format(self.n_samps[i]), buf), dtype=np.float32)
            raw = raw * self.calibrate[i] + self.offset[i]
            data.append(raw)
        return data

    def read_samples(self, channel, beg, end):
        n_samps = self.n_samps[channel]
        begblock = int(floor((beg) / n_samps))
        endblock = int(floor((end) / n_samps))
        data = self.readBlock(begblock)[channel]
        for block in range(begblock+1, endblock+1):
            data = np.append(data, self.readBlock(block)[channel])
        beg -= begblock*n_samps
        end -= begblock*n_samps
        return data[beg:(end+1)]

    def signal_labels(self):
        # convert from unicode to string
        return [str(x) for x in self.ch_names]

    def n_signals(self):
        return self.nchan

    def signal_freqs(self):
        return self.n_samps / self.record_length

    def n_samples(self):
        return self.n_samps * self.n_records

    def read_signal(self, ch_idx):
        beg = 0
        end = self.n_samps[ch_idx] * self.n_records - 1
        return self.read_samples(ch_idx, beg, end)

if __name__ == '__main__':
    with EDF.open(Path("~/Downloads/eeg_recording/ma0844az_1-1+.edf").expanduser()) as edf:
        print(edf.ch_names)
