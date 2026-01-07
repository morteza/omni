---
id: indi_sald
title: Southwest University Adult Lifespan Dataset (SALD)
n_subjects: 494
bids: true
modalities:
  anat: true
  func: true
  rest: true
  eeg: false
scripts:
  download: aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/ INDI_SALD_RAWData_BIDS/
notes: |
  download size: 23.49 GB
---

#
