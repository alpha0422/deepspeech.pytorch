#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Merge nvprof result and plot_nvprof.py result.
"""

import pandas as pd
import getopt
import sys

# Global variable
pf_annote = None
pf_nvprof = None
pf_writer = None

# Parse command line
def parse_cmd_line():
    "Uses getopt to parse cmd line options from user"
    options = ''
    long_options = ['log_file=', 'plot_file=', 'out_file=']

    try:
        opts, extra_args = getopt.gnu_getopt(sys.argv[1:], options,
                                             long_options)
    except:
        raise

    global pf_annote, pf_nvprof, pf_writer

    for opt, arg in opts:
        if opt == '--plot_file':
            pf_annote = pd.read_excel(arg)
        elif opt == '--log_file':
            pf_nvprof = pd.read_csv(arg, skiprows=3, skipfooter=1,
                                    engine='python')
        elif opt == '--out_file':
            pf_writer = pd.ExcelWriter(arg, engine = 'xlsxwriter')
        else:
            raise ValueError

    assert pf_annote is not None
    assert pf_nvprof is not None
    assert pf_writer is not None
    return

# Merge the data frame by correlation id
def merge_frame_by_correlation_id():
    tmp_frame = pf_nvprof.merge(pf_annote[['CorrId', 'LayerName', 'LayerType',
                'Phase']], how='left', left_on='Correlation_ID',
                right_on='CorrId')
    return tmp_frame

if __name__ == '__main__':
    parse_cmd_line()

    tmp_frame = merge_frame_by_correlation_id()

    pf_nvprof.to_excel(pf_writer, sheet_name='Raw Profiling', index=False)
    tmp_frame.to_excel(pf_writer, sheet_name='Annote Layer', index=False)

    pf_writer.save()
    pf_writer.close()

