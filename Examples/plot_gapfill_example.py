import os
import sys
if sys.path[0] != os.getcwd():
    sys.path.insert(0, os.getcwd())
lib_local_path = os.path.normpath(r'C:\WORKSPACE\DEV\gapfill')
if os.path.exists(lib_local_path):
    if sys.path[1] != lib_local_path:
        sys.path.insert(1, lib_local_path)
import numpy as np
import gapfill as gf
import btk

import matplotlib.pyplot as plt
#%%
c3d_sample_dir = os.path.normpath(r'..\Samples_C3D')
src_c3d_path = os.path.join(c3d_sample_dir, r'Sample26\Walking_Hybrid_1_5.c3d')

reader = btk.btkAcquisitionFileReader()
reader.SetFilename(src_c3d_path)
reader.Update()
reader_io = reader.GetAcquisitionIO()
byte_order = reader_io.GetByteOrderAsString()
stor_fmt = reader_io.GetStorageFormatAsString()
f_type = reader_io.GetFileType()
acq = reader.GetOutput()

pts = acq.GetPoints()
num_pts = pts.GetItemNumber()
dict_mkrs = {}
for i in range(num_pts):
    pt = pts.GetItem(i)
    name = pt.GetLabel()
    pos = pt.GetValues()
    resid = pt.GetResiduals().flatten()
    blocked = np.where(np.isclose(resid, -1), True, False)
    pos[blocked,:] = np.nan
    dict_mkrs.update({name: {}})
    dict_mkrs[name].update({'POS': pos})
    dict_mkrs[name].update({'RESID': resid})


tgt_mkr_pos = dict_mkrs['R_SHANK_3']['POS'].copy()
tgt_mkr_pos_original = dict_mkrs['R_SHANK_3']['POS'].copy()

cl_mkr_pos = np.zeros((3, tgt_mkr_pos.shape[0], 3), dtype=np.float32)
for i, mkr_name in enumerate(['R_SHANK_1', 'R_SHANK_2', 'R_SHANK_4']):
    cl_mkr_pos[i] = dict_mkrs[mkr_name]['POS']

# tgt_mkr_pos will be updated.
ret, updated_frs_mask = gf.fill_marker_gap_rbt(tgt_mkr_pos, cl_mkr_pos)

# ret, updated_frs_mask = gf.fill_marker_gap_pattern(tgt_mkr_pos, dict_mkrs['R_SHANK_1']['POS'])

# ret, updated_frs_mask = gf.fill_marker_gap_interp(tgt_mkr_pos)
tgt_mkr_pos_updated = tgt_mkr_pos.copy()
tgt_mkr_pos_updated[~updated_frs_mask] = np.nan

fig = plt.figure(figsize=(4,2))
plt.plot(tgt_mkr_pos_updated[:,0], ls=':', lw=1.5, color='black', zorder=0)
plt.plot(tgt_mkr_pos_original[:,0], ls='-', lw=3, color='red', zorder=1)
plt.plot(tgt_mkr_pos_updated[:,1], ls=':', lw=1.5, color='black', zorder=0)
plt.plot(tgt_mkr_pos_original[:,1], ls='-', lw=3, color='green', zorder=1)
plt.plot(tgt_mkr_pos_updated[:,2], ls=':', lw=1.5, color='black', zorder=0)
plt.plot(tgt_mkr_pos_original[:,2], ls='-', lw=3, color='blue', zorder=1)
plt.show()
# plt.savefig('Sample26_Walking_Hybrid_1_5_RHANK_3.png', dpi=320)