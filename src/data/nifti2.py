from bz2 import compress
from dataclasses import dataclass
from operator import is_
from pathlib import Path
from typing import IO, Optional
import struct
import numpy as np
import gzip
import os
import pyarrow as pa

DEBUG = False


@dataclass(init=False, repr=False)
class NIfTI2:
    """NIfTI-2 reader."""

    # Header fields (v2 spec)
    sizeof_hdr: int
    magic: bytes
    datatype: int  # see https://brainder.org/2012/09/23/the-nifti-file-format/
    bitpix: int
    dim: np.ndarray  # dim[8]
    intent_p1: float
    intent_p2: float
    intent_p3: float
    pixdim: np.ndarray  # pixdim[8]
    vox_offset: int
    scl_slope: float
    scl_inter: float
    cal_max: float
    cal_min: float
    slice_duration: float
    toffset: float
    slice_start: int
    slice_end: int
    descrip: str
    aux_file: str
    qform_code: int
    sform_code: int
    quatern_b: float
    quatern_c: float
    quatern_d: float
    qoffset_x: float
    qoffset_y: float
    qoffset_z: float
    srow_x: np.ndarray  # 4
    srow_y: np.ndarray  # 4
    srow_z: np.ndarray  # 4
    slice_code: int
    xyzt_units: int
    intent_code: int
    intent_name: str
    dim_info: int
    unused_str: bytes

    fname: Path
    is_gzipped: bool

    def __init__(self, fname: Path):
        self.fname = fname
        self.is_gzipped = str(fname).endswith('.gz')
        self.f: IO[bytes] = open(os.devnull, "rb")
        self.f.close()

    @classmethod
    def open(cls, fname):
        self = cls(fname)
        return self

    def __enter__(self):
        try:
            if self.f is None or self.f.closed:
                if self.is_gzipped:
                    self.f = gzip.open(self.fname, "rb")  # type: ignore
                else:
                    self.f = open(self.fname, "rb")
                self.read_header()
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
        """Read NIfTI-2 header (540 bytes)."""
        # NIfTI-2 header structure
        # See: https://www.nitrc.org/docman/view.php/26/1302/nifti2_doc.html

        self.f.seek(0)

        # Read the header (540 bytes for NIfTI-2)
        header_bytes = self.f.read(540)

        # Check if this is NIfTI-1 or NIfTI-2
        sizeof_hdr = struct.unpack('<i', header_bytes[0:4])[0]

        if sizeof_hdr == 348:
            # NIfTI-1 format
            self._read_nifti1_header(header_bytes)
        elif sizeof_hdr == 540:
            # NIfTI-2 format
            self._read_nifti2_header(header_bytes)
        else:
            raise ValueError(f"Invalid NIfTI header size: {sizeof_hdr}. Expected 348 (NIfTI-1) or 540 (NIfTI-2)")

    def _read_nifti1_header(self, header_bytes):
        """Parse NIfTI-1 header (348 bytes)."""
        offset = 0

        # int32 sizeof_hdr (must be 348)
        self.sizeof_hdr = struct.unpack('<i', header_bytes[offset:offset+4])[0]
        offset += 4

        # char data_type[10], db_name[18], extents[4], session_error[2], regular[1] - unused in NIfTI
        offset += 10 + 18 + 4 + 2 + 1

        # char dim_info
        self.dim_info = struct.unpack('B', header_bytes[offset:offset+1])[0]
        offset += 1

        # short dim[8]
        self.dim = np.array(struct.unpack('<8h', header_bytes[offset:offset+16]))
        offset += 16

        # float intent_p1, intent_p2, intent_p3
        self.intent_p1, self.intent_p2, self.intent_p3 = struct.unpack('<3f', header_bytes[offset:offset+12])
        offset += 12

        # short intent_code
        self.intent_code = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # short datatype
        self.datatype = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # short bitpix
        self.bitpix = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # short slice_start
        self.slice_start = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # float pixdim[8]
        self.pixdim = np.array(struct.unpack('<8f', header_bytes[offset:offset+32]))
        offset += 32

        # float vox_offset
        self.vox_offset = int(struct.unpack('<f', header_bytes[offset:offset+4])[0])
        offset += 4

        # float scl_slope, scl_inter
        self.scl_slope, self.scl_inter = struct.unpack('<2f', header_bytes[offset:offset+8])
        offset += 8

        # short slice_end
        self.slice_end = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # char slice_code
        self.slice_code = struct.unpack('B', header_bytes[offset:offset+1])[0]
        offset += 1

        # char xyzt_units
        self.xyzt_units = struct.unpack('B', header_bytes[offset:offset+1])[0]
        offset += 1

        # float cal_max, cal_min
        self.cal_max, self.cal_min = struct.unpack('<2f', header_bytes[offset:offset+8])
        offset += 8

        # float slice_duration
        self.slice_duration = struct.unpack('<f', header_bytes[offset:offset+4])[0]
        offset += 4

        # float toffset
        self.toffset = struct.unpack('<f', header_bytes[offset:offset+4])[0]
        offset += 4

        # int32 glmax, glmin - unused
        offset += 8

        # char descrip[80]
        self.descrip = header_bytes[offset:offset+80].split(b'\x00')[0].decode('latin-1')
        offset += 80

        # char aux_file[24]
        self.aux_file = header_bytes[offset:offset+24].split(b'\x00')[0].decode('latin-1')
        offset += 24

        # short qform_code, sform_code
        self.qform_code, self.sform_code = struct.unpack('<2h', header_bytes[offset:offset+4])
        offset += 4

        # float quatern_b, quatern_c, quatern_d
        self.quatern_b, self.quatern_c, self.quatern_d = struct.unpack('<3f', header_bytes[offset:offset+12])
        offset += 12

        # float qoffset_x, qoffset_y, qoffset_z
        self.qoffset_x, self.qoffset_y, self.qoffset_z = struct.unpack('<3f', header_bytes[offset:offset+12])
        offset += 12

        # float srow_x[4], srow_y[4], srow_z[4]
        self.srow_x = np.array(struct.unpack('<4f', header_bytes[offset:offset+16]))
        offset += 16
        self.srow_y = np.array(struct.unpack('<4f', header_bytes[offset:offset+16]))
        offset += 16
        self.srow_z = np.array(struct.unpack('<4f', header_bytes[offset:offset+16]))
        offset += 16

        # char intent_name[16]
        self.intent_name = header_bytes[offset:offset+16].split(b'\x00')[0].decode('latin-1')
        offset += 16

        # char magic[4]
        self.magic = header_bytes[offset:offset+4]
        offset += 4

        # Set defaults for NIfTI-2 fields not in NIfTI-1
        self.unused_str = b''

        # Apply default scaling if not set
        # Apply default scaling if not set
        if self.scl_slope == 0 or np.isnan(self.scl_slope):
            self.scl_slope = 1.0
        if self.scl_inter is None or np.isnan(self.scl_inter):
            self.scl_inter = 0.0

    def _read_nifti2_header(self, header_bytes):
        """Parse NIfTI-2 header (540 bytes)."""
        offset = 0

        # int32 sizeof_hdr (must be 540)
        self.sizeof_hdr = struct.unpack('<i', header_bytes[offset:offset+4])[0]
        offset += 4

        # char magic[8]
        self.magic = header_bytes[offset:offset+8]
        offset += 8

        # int16 datatype
        self.datatype = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # int16 bitpix
        self.bitpix = struct.unpack('<h', header_bytes[offset:offset+2])[0]
        offset += 2

        # int64 dim[8]
        self.dim = np.array(struct.unpack('<8q', header_bytes[offset:offset+64]))
        offset += 64

        # double intent_p1, intent_p2, intent_p3
        self.intent_p1, self.intent_p2, self.intent_p3 = struct.unpack('<3d', header_bytes[offset:offset+24])
        offset += 24

        # double pixdim[8]
        self.pixdim = np.array(struct.unpack('<8d', header_bytes[offset:offset+64]))
        offset += 64

        # double vox_offset (stored as float64 in NIfTI-2)
        self.vox_offset = struct.unpack('<d', header_bytes[offset:offset+8])[0]
        offset += 8

        # double scl_slope, scl_inter
        self.scl_slope, self.scl_inter = struct.unpack('<2d', header_bytes[offset:offset+16])
        offset += 16

        # double cal_max, cal_min
        self.cal_max, self.cal_min = struct.unpack('<2d', header_bytes[offset:offset+16])
        offset += 16

        # double slice_duration
        self.slice_duration = struct.unpack('<d', header_bytes[offset:offset+8])[0]
        offset += 8

        # double toffset
        self.toffset = struct.unpack('<d', header_bytes[offset:offset+8])[0]
        offset += 8

        # int64 slice_start, slice_end
        self.slice_start, self.slice_end = struct.unpack('<2q', header_bytes[offset:offset+16])
        offset += 16

        # char descrip[80]
        self.descrip = header_bytes[offset:offset+80].split(b'\x00')[0].decode('latin-1')
        offset += 80

        # char aux_file[24]
        self.aux_file = header_bytes[offset:offset+24].split(b'\x00')[0].decode('latin-1')
        offset += 24

        # int32 qform_code, sform_code
        self.qform_code, self.sform_code = struct.unpack('<2i', header_bytes[offset:offset+8])
        offset += 8

        # double quatern_b, quatern_c, quatern_d
        self.quatern_b, self.quatern_c, self.quatern_d = struct.unpack('<3d', header_bytes[offset:offset+24])
        offset += 24

        # double qoffset_x, qoffset_y, qoffset_z
        self.qoffset_x, self.qoffset_y, self.qoffset_z = struct.unpack('<3d', header_bytes[offset:offset+24])
        offset += 24

        # double srow_x[4], srow_y[4], srow_z[4]
        self.srow_x = np.array(struct.unpack('<4d', header_bytes[offset:offset+32]))
        offset += 32
        self.srow_y = np.array(struct.unpack('<4d', header_bytes[offset:offset+32]))
        offset += 32
        self.srow_z = np.array(struct.unpack('<4d', header_bytes[offset:offset+32]))
        offset += 32

        # int32 slice_code
        self.slice_code = struct.unpack('<i', header_bytes[offset:offset+4])[0]
        offset += 4

        # int32 xyzt_units
        self.xyzt_units = struct.unpack('<i', header_bytes[offset:offset+4])[0]
        offset += 4

        # int32 intent_code
        self.intent_code = struct.unpack('<i', header_bytes[offset:offset+4])[0]
        offset += 4

        # char intent_name[16]
        self.intent_name = header_bytes[offset:offset+16].split(b'\x00')[0].decode('latin-1')
        offset += 16

        # char dim_info
        self.dim_info = struct.unpack('B', header_bytes[offset:offset+1])[0]
        offset += 1

        # char unused_str[15]
        self.unused_str = header_bytes[offset:offset+15]
        offset += 15

        # Apply default scaling if not set
        if self.scl_slope == 0 or np.isnan(self.scl_slope):
            self.scl_slope = 1.0
        if self.scl_inter is None or np.isnan(self.scl_inter):
            self.scl_inter = 0.0

    def _get_dtype(self):
        """Get numpy dtype from NIfTI datatype code."""
        # NIfTI datatype codes
        dtype_map = {
            2: np.uint8,     # UINT8
            4: np.int16,     # INT16
            8: np.int32,     # INT32
            16: np.float32,  # FLOAT32
            64: np.float64,  # FLOAT64
            256: np.int8,    # INT8
            512: np.uint16,  # UINT16
            768: np.uint32,  # UINT32
            1024: np.int64,  # INT64
            1280: np.uint64, # UINT64
        }

        if self.datatype not in dtype_map:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

        return dtype_map[self.datatype]

    def get_shape(self):
        """Get the shape of the data array."""
        ndim = self.dim[0]
        return tuple(self.dim[1:ndim+1])

    def get_affine(self):
        """Get the affine transformation matrix (4x4)."""
        # Prefer sform over qform
        if self.sform_code > 0:
            affine = np.eye(4)
            affine[0, :] = self.srow_x
            affine[1, :] = self.srow_y
            affine[2, :] = self.srow_z
            return affine
        elif self.qform_code > 0:
            # Build affine from quaternion (simplified version)
            # For full implementation, see nibabel's quaternion handling
            affine = np.eye(4)
            affine[0, 3] = self.qoffset_x
            affine[1, 3] = self.qoffset_y
            affine[2, 3] = self.qoffset_z
            # Apply pixdim scaling
            affine[0, 0] = self.pixdim[1]
            affine[1, 1] = self.pixdim[2]
            affine[2, 2] = self.pixdim[3]
            return affine
        else:
            # No spatial transform specified, use pixdim
            affine = np.eye(4)
            affine[0, 0] = self.pixdim[1]
            affine[1, 1] = self.pixdim[2]
            affine[2, 2] = self.pixdim[3]
            return affine

    def read_data(self, volume_idx=None):
        """
        Read a single volume from a 4D dataset.
        For 3D data, volume_idx is ignored.
        """
        dtype = self._get_dtype()
        shape = self.get_shape()

        if volume_idx is None or len(shape) == 3:
            # 3D data, just read the whole thing
            n_voxels = np.prod(shape)
            offset = int(self.vox_offset)
        elif len(shape) == 4:
            # 4D data, read one specific volume
            n_voxels = np.prod(shape[:3])
            offset = int(self.vox_offset + volume_idx * n_voxels * dtype().itemsize)
        else:
            raise ValueError(f"Unsupported dimensionality: {len(shape)}D")

        self.f.seek(offset)
        data = np.frombuffer(self.f.read(int(n_voxels * dtype().itemsize)), dtype=dtype)
        data = data.reshape(shape, order='F')

        # Apply scaling
        if self.scl_slope != 1.0 or self.scl_inter != 0.0:
            data = data.astype(np.float32)
            data = data * self.scl_slope + self.scl_inter

        return data

    def n_volumes(self):
        """Return the number of volumes (for 4D data) or 1 (for 3D data)."""
        shape = self.get_shape()
        if len(shape) >= 4:
            return shape[3]
        return 1

    def get_voxel_size(self):
        """Get the voxel sizes in mm."""
        ndim = self.dim[0]
        return tuple(self.pixdim[1:ndim+1])


    def to_arrow_tensor(self, volume_idx: Optional[int] = None) -> pa.Tensor:
        """Convert the loaded NIfTI data (or a specific volume) to an Apache Arrow DenseTensor."""
        data = self.read_data(volume_idx)

        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        return pa.Tensor.from_numpy(data)


def profile_custom_nifti2():
    
    import time
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    with NIfTI2.open(test_file) as custom:
        data = custom.read_data()

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("[PROFILE] CUSTOM")
    print(data.shape, data.dtype)
    print(f"Time: {elapsed:.2f}s")
    print(f"Curr mem: {current / 1e6:.1f} MB")
    print(f"Peak mem: {peak / 1e6:.1f} MB")
    print(f"Diff mem: {(peak - current) / 1e6:.1f} MB")

    return data

def profile_nibabel():
    import nibabel as nib  # type: ignore

    import time
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    data = nib.load("~/workspace/omni/tmp/rest-7T.nii.gz").get_fdata(dtype=np.float32)  # type: ignore[attr-defined]

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("[PROFILE] NIBABEL")
    print(data.shape, data.dtype)
    print(f"Time: {elapsed:.2f}s")
    print(f"Curr mem: {current / 1e6:.1f} MB")
    print(f"Peak mem: {peak / 1e6:.1f} MB")
    print(f"Diff mem: {(peak - current) / 1e6:.1f} MB")

    # print(f"nibabel shape: {nib_data.shape}, dtype: {nib_data.dtype}")
    # print(f"nibabel range: [{nib_data.min():.2f}, {nib_data.max():.2f}]")
    # print(f"nibabel mean: {nib_data.mean():.2f}")

    return data

if __name__ == '__main__':
    import sys

    # Test with the sample file (change path as needed)
    test_file = Path("~/workspace/omni/tmp/rest-7T.nii.gz").expanduser()

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])

    if not test_file.exists():
        print(f"File not found: {test_file}")
        sys.exit(1)

    print(f"Reading NIfTI file: {test_file}")

    with NIfTI2.open(test_file) as nii:
        print(f"NIfTI version: NIfTI-{1 if nii.sizeof_hdr == 348 else 2}")
        print(f"Magic: {nii.magic}")
        print(f"Data type: {nii.datatype} (bitpix: {nii.bitpix})")
        print(f"Dimensions: {nii.get_shape()}")
        print(f"Voxel sizes: {nii.get_voxel_size()} mm")
        print(f"Description: {nii.descrip}")
        print(f"Aux file: {nii.aux_file}")
        print(f"Data offset: {nii.vox_offset}")
        print(f"Scaling: slope={nii.scl_slope}, intercept={nii.scl_inter}")
        print(f"Calibration: min={nii.cal_min}, max={nii.cal_max}")
        print(f"Number of volumes: {nii.n_volumes()}")
        print(f"Intent: code={nii.intent_code}, name={nii.intent_name}")
        print(f"Intent params: p1={nii.intent_p1}, p2={nii.intent_p2}, p3={nii.intent_p3}")
        print(f"Slice: start={nii.slice_start}, end={nii.slice_end}, code={nii.slice_code}, duration={nii.slice_duration}")
        print(f"Time offset: {nii.toffset}")
        print(f"Qform code: {nii.qform_code}")
        print(f"Sform code: {nii.sform_code}")
        print(f"Quaternion: b={nii.quatern_b}, c={nii.quatern_c}, d={nii.quatern_d}")
        print(f"Qoffset: x={nii.qoffset_x}, y={nii.qoffset_y}, z={nii.qoffset_z}")
        print(f"Units: xyzt={nii.xyzt_units}")
        print(f"Dim info: {nii.dim_info}")
        print()
        print("Affine matrix:")
        print(nii.get_affine())
        print()
        print("Srow_x:", nii.srow_x)
        print("Srow_y:", nii.srow_y)
        print("Srow_z:", nii.srow_z)

        # Read data
        data = nii.read_data()

        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

        print(f"Data mean: {data.mean():.2f}")


        # print(f"Writing DenseTensor to {out_path} with {compression.upper()} compression...")
        # compression = "zstd"
        # compression_level = 9  # higher = better ratio, slower
        # out_path = Path(f"nifti_tensor.arrow.{compression}").resolve()
        # # Write tensor with compression
        # with pa.CompressedOutputStream(out_path, compression, compression_level=compression_level) as sink:
        #     pa.ipc.write_tensor(tensor, sink)

        data_tensor = nii.to_arrow_tensor()
        print(f"Converted to Apache Arrow DenseTensor with shape {data_tensor.shape} and dtype {data_tensor.type}")

        import zarr
        z = zarr.create_array(
            Path("tmp/tensor6.zarr"),
            shape=data.shape,
            chunks=(32, 32, 32, data.shape[-1]),
            # dtype=data_tensor.type.to_pandas_dtype(),
            dtype=data.dtype,
            compressors=[zarr.codecs.ZstdCodec(level=9)],
            overwrite=True,
        )
        z[:] = data # data_tensor.to_numpy()  # streams chunk-by-chunk    
        print("Converted to zarr.")
        print(f"Data mean from zarr: {np.array(z).mean():.2f}")

        print()
        data_nib = profile_nibabel()
        print()
        data_cus = profile_custom_nifti2()

        print()
        print("avg diff values:", np.abs(data_nib - data_cus).mean())
        print("max diff values:", np.abs(data_nib - data_cus).max())
