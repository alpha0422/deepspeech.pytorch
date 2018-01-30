################################################################################
##
##  plot_nvprof
##
##                 - Automatically generate a python script
##                 - Script will open files for reading and writing
##                 - And will have examples of basic operations like formatted
##                 - printing
##
##  asettle
##  Fri Nov 3 2017
##
################################################################################
from __future__ import print_function
import getopt
import sys
import sqlite3
import re
import os
import pandas            as pd
import numpy             as np
#import matplotlib.pyplot as plt
import fancydebug

################################################################################
## Algorithm to Link GPU events to Layer names
##
##  For each GPU event in CONCURRENT_KERNELS
##  Use the correlation ID to map the GPU event to the runtime cuda event (function call)
##  Now for this Runtime event - record the start time and end time
##  Then Go to the markers - find all markers whose start times are > runtime event
##  start and whose end time is > runtime event end
##   There should be 1 Marker that meets this criteria
################################################################################
################################################################################
## Global Variables
################################################################################
## GetOpt set up
options      = 'h'     ## Help message - string of possible 1 char options, ':' after option means it takes an arg
long_options      = ['in_files=', 'out_file=', 'debug', 'help'] ## List of long form options
db_file_list      = []    ## Input file seql DB
pivot_tbl         = None    ## Output pivot table
excel_file_name   = None
excel_writer      = None   ## Pandas excel file writing handle
string_hash       = {}     ## Hash table - maps string ID to name
time_base         = -1     ## Starting time stamp of experiment
Debug             = False  ## True for print debugging
max_int32         = 1 << 32 ## Max 32 bit val
################################################################################
## Function definitions
################################################################################
################################################################################
##
##  usage()
##
##    print a help message then exit 0
##
################################################################################
def usage ():
    "Print a help message then exit"
    print("Usage: plot_nvprof [-h] --in_files nvp_sqlite_file,nvp_file1,nvp_file2 -out_file output_file_name [--tag pivot_tbl_field] [--debug]", end='\n')
    sys.exit (0)

################################################################################
##
## parse_cmd_line()
##
##   Uses getopt to parse cmd line options from user
##
################################################################################
def parse_cmd_line () :
    "Uses getopt to parse cmd line options from user"
    ## Exception handling
    try:
        opts, extra_args = getopt.gnu_getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError as err :
        print ("Exception caught :  {0}".format(err))    ## Didn't specify type of err in format specifier
        sys.exit(1)

    ## Mark this as global scope because other functions need this value
    global db_file_list
    global Debug
    global pivot_tbl
    global excel_file_name
    global excel_writer

    ## Walk list of cmd line options - opts is a pair<string,string>
    for opt, arg in opts:
        if (opt == "-i" or opt == "--in_files"):
            db_file_list = re.split(',', arg)
            print ("Reading in_file {0}".format(arg))
        elif (opt == "-o" or opt == "--out_file"):
            print ("Writing out file {0:s}".format(arg))
            pivot_tbl = arg
            excel_file_name = re.sub(r'.txt', r'.xlsx', pivot_tbl)
            excel_writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
        elif (opt == "-h" or opt == "--help"):
            usage();
        elif (opt == "-d" or opt == "--debug") :
            print("Enabling Debug print messages")
            Debug = True
    return
################################################################################
##
## open_output_files
################################################################################
def open_ouput_file() :
    """
    Check to see if output file specified on cmd line, else use stdout
    """
    global pivot_tbl
    ## Open the output file (pivot_table)
    if(pivot_tbl is None) :
        file_des = sys.stdout
    else :
        file_des = open(pivot_tbl, "w")
    return file_des

################################################################################
##
## run_io_examples
##
##   File io examples
################################################################################
def run_io_examples ():
    "Examples of print usage"
    ## Open a file for reading and write to it
    file_des = open("tmp.txt", "w")

    ## unformatted print
    print ("## Example python file writes", end="\n", file=file_des)
    print ("## Unformatted prints", end="\n", file=file_des)
    print (65, "F", sep="...", end="\n", file=file_des)
    print ((65 -32 ) * 5 / 9, "C", sep=" ?", end="\n", file=file_des)
    return

################################################################################
##
## reset_global_vars
################################################################################
def reset_global_vars() :
    global string_hash
    time_base = -1
    del string_hash
    string_hash = {}

    return

################################################################################
##
## read_db_file
##
##   Read in database
################################################################################
def read_db_file (db_name=None, output_fd=None):
    "Read in the DB file and extract relevant tables"
    if db_name is None or output_fd is None:
        print("Error read_db_file: No db file specified - exiting. ")
        sys.exit(1)
    print ("Reading DB file {0}".format(db_name), end='\n')
    connection = sqlite3.connect(db_name)
    cur        = connection.cursor();
    cur.execute("select name from sqlite_master where type='table'")
    #dump_cur(cur)
    all_tbls = get_tbl_names(cur)
    remaining_tbls = []

    ## Read in StringTable and DRIVER first to extract global info used by other table processing
    for tbl in all_tbls:
        update_list = 1
        for tbl_type in ['DRIVER', 'StringTable']:
            pattern = re.compile(tbl_type)
            if pattern.search(tbl) :
                process_tbl(tbl, cur, tbl_type)
                update_list = 0
        if update_list :
            remaining_tbls.append(tbl)

    # Walk the remaining list of tables
    if(Debug) :
        for tbl in remaining_tbls:
            print ("Tbl {0:s}".format(tbl), end='\n')
            #process_runtime_tbl(tbl, cur)
            tbl_str = re.sub(r".*_KIND_", "", tbl)
            print ("tbl str {0} from table {1}". format(tbl_str, tbl))
            process_tbl(tbl, cur, tbl_str)

    ## Layer names (CPU Runtime) to the kernels (GPU) that they launch
    panda_frame = link_kernel_to_dl_layer(cur, all_tbls, db_name, output_fd)
    connection.close()

    ## Clear globals that are set up on each pass of the db file
    reset_global_vars()

    return panda_frame

################################################################################
## dump_rows()
##
##   Walk all the rows in the table and print
################################################################################
def dump_rows(cursor=None, tbl_hdr=None, tbl_type=None):
    if cursor is None:
        print ("Error dump_rows: No cursor specified - exiting.")
        sys.exit(1)

    if tbl_hdr is None:
        print ("Error dump_rows_runtime: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_runtime: No table type name specified- exiting.")
        sys.exit(1)

    ## Check the tbl_type - call the tbl specific dump function
    if (tbl_type == 'RUNTIME') or (tbl_type == 'DRIVER') :
        dump_rows_runtime_driver(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'NAME' :
        dump_rows_name(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'StringTable' :
        dump_rows_strings(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'MARKER' :
        dump_rows_marker(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'CONCURRENT_KERNEL' :
        dump_rows_conc_kernel(cursor, tbl_hdr, tbl_type)
    else:
        dump_rows_default(cursor, tbl_hdr, tbl_type)

    return
################################################################################
## dump_rows_default()
################################################################################
def dump_rows_default (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type NAME "
    if cur is None:
        print ("Error dump_rows_default: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_default: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_default: No table type name specified - exiting.")
        sys.exit(1)

    for row in cur:
        if Debug:
            print ("DEFAULT {0} {1}".format(tbl_type, row))

    return
################################################################################
## dump_rows_name()
################################################################################
def dump_rows_name (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type NAME "
    if cur is None:
        print ("Error dump_rows_name: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_name: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_name: No table type name specified - exiting.")
        sys.exit(1)

    # Get Row indexes
    if ('objectKind' in hdr)  and ('objectId' in hdr) and ('name')  :
        obj_kind_idx    = hdr['objectKind']
        obj_id_idx      = hdr['objectId']
        name_idx        = hdr['name']
    else :
        print ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
        sys.exit(1)

    for row in cur:
        if Debug :
            print ("{0} {1} {2} {3}".format(tbl_type, row[name_idx], row[obj_kind_idx], row[obj_id_idx]))

    return
################################################################################
## dump_rows_strings()
################################################################################
def dump_rows_strings(cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type StringTable "
    if cur is None:
        print ("Error dump_rows_strings: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_strings: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_strings: No table type name specified - exiting.")
        sys.exit(1)

    if ('_id_' in hdr)  and ('value' in hdr)  :
        str_id_idx   = hdr['_id_']
        str_name_idx = hdr['value']

    for row in cur:
        str_id   = row[str_id_idx]
        str_name = row[str_name_idx]
        if str_id not in string_hash:
            string_hash[str_id] = str_name
        if Debug:
            print ("{0} {1} {2}".format(tbl_type, row[str_id_idx], row[str_name_idx]))

    return

################################################################################
## dump_rows_conc_kernel()
##
##  Note that the correlation ID in conc kernel maps to correlation ID in Runtime
##  Not always true in the reverse direction - Runtime covers more events
##  than just kernel
################################################################################
def dump_rows_conc_kernel(cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type CONCURRENT_KERNEL "
    global time_base
    # Get Row indexes
    if ('start' in hdr)  and ('end' in hdr) and ('registersPerThread' in hdr) and ('name' in hdr) and ('correlationId') and ('streamId')  :
        start_idx          = hdr['start']
        end_idx            = hdr['end']
        corr_id_idx        = hdr['correlationId']
        name_id_idx        = hdr['name']
        stream_id_idx      = hdr['streamId']
        regs_per_th_idx    = hdr['registersPerThread']
    else :
        print ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
        sys.exit(1)

    if Debug :
        print ("TblType ElapsedTime(ns) StartTime(ns) EndTime(ns) StreamId CorrId Regs Name")
    for row in cur:
        name_id         = row[name_id_idx]
        start_time      = row[start_idx]
        end_time        = row[end_idx]
        string_name     = string_hash[name_id]
        ## Get the first time stamp so we can subtract off the time since epoc
        if time_base == -1:
            time_base = start_time

        if Debug :
            print ("{0} {1} {2} {3} {4} {5} {6} {7}".format(tbl_type, end_time - start_time, start_time - time_base, end_time - time_base,  row[stream_id_idx], row[corr_id_idx], row[regs_per_th_idx], string_name))

    return
################################################################################
## dump_rows_marker()
##
## Format for this table is 2 lines per event
##  First row - time stamp is the start time and the 'name' field is the string name of the event
##  Use the String Table to lookup the names - name to ID mapping - only valid for start of event row
##  The 'id' col is the event ID and it should be the same for both rows
##   2nd Row - Time stamp is stop time
##    Use 'id' to match up the start time stamp and event info
##  Additional info is available in the marker_data() table - use 'id' to lookup this data
##  'Category' is the field that is reported by the GUI
##  _id_,flags,timestamp,id,objectKind,objectId,name,domain
#    1,2,1509565664581882230,1,2,"^Z",3,0
#    2,4,1509565664620622854,1,2,"^Z",0,0

################################################################################
def dump_rows_marker (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type MARKER "

    global time_base
    marker_hash = {}
    if cur is None:
        print ("Error dump_rows_marker: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_marker: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_marker: No table type name specified - exiting.")
        sys.exit(1)

    # Get Row indexes
    if ('timestamp' in hdr)  and ('flags' in hdr) and ('id' in hdr) and ('name' in hdr)  :
        ts_idx          = hdr['timestamp']
        flag_idx        = hdr['flags']
        event_id_idx    = hdr['id']
        name_id_idx     = hdr['name']
    else :
        print ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
        sys.exit(1)

    if Debug:
        print ("TblType EventId NameId ElapsedTime(ns) StartTime(ns) EndTime(ns) LayerName LayerInstance")
    for row in cur:
        if time_base == -1:
            time_base = row[ts_idx]
        event_id  = row[event_id_idx]
        ## Save the name_id and the start time stamp for each event
        if event_id not in marker_hash :
            marker_hash[event_id] = [row[name_id_idx], row[ts_idx]]
            #print ("Adding event_id {0} to marker hash".format(event_id))
        else :
            name_id, start_time = marker_hash[event_id]
            elapsed_time = row[ts_idx] - start_time ## Elapsed time in ns
            string_net_name     = string_hash[name_id]
            net_name, long_name = string_net_name.split(' ')
            if Debug:
                print ("{0} {1} {2} {3} {4} {5} {6} {7}".format(tbl_type, event_id, name_id, elapsed_time, start_time - time_base, row[ts_idx] - time_base, net_name, long_name))
            if (row[flag_idx] !=4) :
                print ("Error - unexpected flag {0} for row {1}".format(row[flag_idx], row))
            del (marker_hash[event_id])

    return

################################################################################
## dump_rows_runtime()
##
##   Walk all the rows in the table and print
##  runtime events map to different tables
##    Many events in runtime are cuda events - use the correlation ID to
##   lookup the CUDA event ID in the the table CUDA_EVENT
##   The events are numbered - I don't see a string equivalent to the number
##   The profiler must have an internal decoder for these events
##  - The other type of event is kernel event
##  - These events map to a different table
##    - So if the correlation ID is not found in cuda_event table
##    - Look in concurrent Kernel event table
##    - If the correlation ID matches - then check the Name ID field
##    - The name ID should return the string name of the event
##     - You can also compare time stamp info because the kernel table tracks it
################################################################################
def dump_rows_runtime_driver (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for TBL type RUNTIME or driver"
    global time_base
    if cur is None:
        print ("Error dump_rows_runtime_driver: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_runtime_driver: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_runtime_driver: No tbl type name specified - exiting.")
        sys.exit(1)

    # Get start time stamp
    if ('start' in hdr)  and ('end' in hdr) and ('threadId') and ('correlationId') and ('cbid')  :
        start_idx   = hdr['start']
        end_idx     = hdr['end']
        thread_idx  = hdr['threadId']
        corr_idx    = hdr['correlationId']
        cb_idx      = hdr['cbid']
    else :
        print ("Error: Col Hdrs {}", format(hdr))
        sys.exit(1)

    # Walk the cursor - print each row
    if Debug:
        print ("Start_time(ns) End_time(ns) Elapsed_time(ns) Thread_id Correlation_id Cb_id")
    for row in cur:
        if time_base == -1:
            time_base = row[start_idx]
        thread = row[thread_idx]
        # For integer values that are < max_int32 and have non zero bit 31 they got converted to negative number by
        #  this equation:  value - max_int32 = new_value (negative number)
        # This code converts the negative number back to the positive int it is supposed to be : pos_int = neg_int + max_int32
        if thread < 0 :
            thread = max_int32 + thread
        if Debug:
            print ("{0} {1} {2} {3} {4} {5} {6}".format(tbl_type, row[start_idx]-time_base, row[end_idx] - time_base, row[end_idx] - row[start_idx], thread, row[corr_idx], row[cb_idx]))

    return

################################################################################
## get_tbl_names()
##
##   Walk all the rows in the table and print
################################################################################
def get_tbl_names (cur=None):
    "Dump the contents of the sql cursor"
    tbl_list = []
    if cur is None:
        print ("Error get_tbl_names: No cursor specified - exiting.\n")
        sys.exit(1)
    for row in cur:
        tbl_name = row[0]
        if Debug :
            print ("Tbl Name {0:s}".format(tbl_name), end='\n')
        tbl_list.append(tbl_name)

    return tbl_list

################################################################################
## get_tbl_hdrs()
################################################################################
def get_tbl_hdrs(cursor=None, display=True):
    tbl_hdr = {}   ## Hash table to map col header to index
    for idx, col in enumerate (cursor.description) :
        if(display) :
            print ("Col Header: {0} index {1}".format(col[0], idx), end='\n')
        tbl_hdr[col[0]] = idx
    if(display) :
        ## Prtint the header in 1 row
        for idx, col in enumerate (cursor.description) :
            print ("{0} ".format(col[0]), end='')
        print ("")
    return tbl_hdr
################################################################################
## process_driver_tbl()
##
##   Decode the DRIVER table
################################################################################
def process_tbl(tbl=None, cur=None, name=None):
    if tbl is None:
        print ("Error process_tbl: No tbl specified - exiting.\n")
        sys.exit(1)
    if cur is None:
        print ("Error process_tbl: No cursor specified - exiting.\n")
        sys.exit(1)
    if name is None:
        print ("Error process_tbl: No name specified - exiting.\n")
        sys.exit(1)

    pattern = re.compile(name)
    if pattern.search(tbl) :
        cmd_string = "select * from {};".format(tbl)
        if Debug:
            print ("Executing sql cmd {}".format(cmd_string), end='\n')
        cur.execute(cmd_string)   ## Need to use a tuple for variable sub- even though only passing 1 value
        tbl_hdr = get_tbl_hdrs(cur, Debug)
        dump_rows(cur, tbl_hdr, name)

################################################################################
## get_marker_pandas_tbl_frame()
##
################################################################################
def get_marker_pandas_tbl_frame(tbl=None, cur=None) :
    """
    Returns pandas tbl frame for the marker table
    """
    query_string = "select name, id, timestamp from {}".format(tbl)
    tbl_hash     = {'name': [], 'name_id': [] , 'id': [], 'timestamp': [] }
    cur.execute(query_string)
    tbl_list    = cur.fetchall()
    tbl_hdr     = get_tbl_hdrs(cur, False)
    name_id_idx = tbl_hdr['name']
    id_idx      = tbl_hdr['id']
    time_idx    = tbl_hdr['timestamp']

    for row in tbl_list:
        marker_name         = string_hash[row[name_id_idx]]
        tbl_hash['name_id'].append(row[name_id_idx])
        tbl_hash['name'].append(marker_name)
        tbl_hash['id'].append(row[id_idx])
        tbl_hash['timestamp'].append(row[time_idx])

    panda_frame = pd.DataFrame(tbl_hash)
    del tbl_hash
    #tmp_frame = panda_frame
    return panda_frame


################################################################################
## get_runtime_pandas_tbl_frame()
##
################################################################################
def get_runtime_pandas_tbl_frame(tbl=None, cur=None) :
    """
    Copy a sql TBL into Pandas frame
    """
    tbl_hash     = {'start': [] , 'end': [], 'threadId': [], 'correlationId': [] }
    query_string = "select start, end, threadId, correlationId from {} ".format(tbl)
    cur.execute(query_string)
    tbl_list    = cur.fetchall()
    tbl_hdr     = get_tbl_hdrs(cur, False)
    start_idx   = tbl_hdr['start']
    end_idx     = tbl_hdr['end']
    th_idx      = tbl_hdr['threadId']
    cor_idx     = tbl_hdr['correlationId']
    for row in tbl_list :
        tbl_hash['start'].append(row[start_idx])
        tbl_hash['end'].append(row[end_idx])
        tbl_hash['threadId'].append(row[th_idx])
        tbl_hash['correlationId'].append(row[cor_idx])

    panda_frame = pd.DataFrame(tbl_hash)
    del tbl_hash
    return panda_frame

################################################################################
## name_lookup_by_id()
##
################################################################################
def link_kernel_to_dl_layer(cur=None, tbl_list=None, db_name=None, file_des=None) :
    """
    Walks the list of GPU kernel events and maps them to user level layer names
    defined in CPU CUDA runtime threads
    """
    if cur is None  or tbl_list is None or db_name is None or file_des is None:
        print ("Error link_kernel_to_dl_layer: bad arguments - exiting.\n")
        sys.exit(1)

    kernel_events = []   ## Empty list - used to store the entire kernel tbl
    tbl_str       = 'CONCURRENT_KERNEL'
    kernel_tbl    = get_tbl_name_from_type(tbl_str, tbl_list)
    tbl_str       = 'RUNTIME'
    runtime_tbl   = get_tbl_name_from_type(tbl_str, tbl_list)
    tbl_str       = 'MARKER'
    marker_tbl    = get_tbl_name_from_type(tbl_str, tbl_list)
    if kernel_tbl is None or runtime_tbl is None or marker_tbl is None:
        print ("Error - Can't find table with substr {0:s} found".format(tbl_str))
        sys.exit(1)
    pivot_tbl_tag   = re.sub(r'[.]\w+', '', db_name)
    #query_string  = "select * from {}".format(kernel_tbl)
    query_string  = "select correlationId, start, end, name from {}".format(kernel_tbl)
    cur.execute(query_string)

    ## Store the whole table in memory
    kernel_events     = cur.fetchall()
    ## Get the runtime table and store in pandas frame
    runtime_tbl_frame = get_runtime_pandas_tbl_frame(runtime_tbl, cur)
    marker_tbl_frame  = get_marker_pandas_tbl_frame(marker_tbl, cur)

    ## Store the table in a dict - Col headers are the keys - each val is a list, then pass the dict to Pandas to make
    ## a frame Walk each row in the table - query the RUNTIME tbl for CPU start/end times
    report_tbl = {'LayerName' : [], 'LayerType' : [], 'Phase' : [],  'CPUStartTime(ns)' : [], 'CPUEndTime(ns)' : [], 'CPUDuration(ns)' : [], 'GPUStartTime(ns)' : [], 'GPUEndTime(ns)' : [], 'GPUDuration(ns)' : [], 'CorrId' : [], 'Thread' : [], 'Kernel' : [], 'ExperTag' : []}
    print ("LayerName LayerType Phase CPUStartTime(ns) CPUEndTime(ns) CPUDuration(ns) GPUStartTime(ns) GPUEndTime(ns) GPUDuration(ns) CorrId Thread Kernel ExperTag", file=file_des)
    for kernel in kernel_events :
        ## Get the correlation ID
        ## Col names match the order used by query_string
        [corr_id, start_time, end_time, name_id] = kernel
        ## Need to call map from name_id to name
        ker_name   = string_hash[name_id]
        ker_name   = demangle_kernel_name(ker_name)
        # Use correlation ID to map kernel event to runtime cpu event
        try:
            cpu_start, cpu_end, thread_id         = get_tbl_event_by_corr_id(corr_id, runtime_tbl_frame)
        except IndexError:
            # wkong modified for kernelPointwiseApply::I::CopyOpI6
            print("Unresolved kernel: {}({})".format(ker_name, corr_id))
        ## Now find marker / range whose start time > cpu_start and end_time > cpu_end
        marker_name, marker_start, marker_end = get_tbl_marker_by_time_window(cpu_start, cpu_end, marker_tbl_frame)

        [phase, layer_name] = marker_name.split(' ')
        layer_type          = get_layer_type_from_name(layer_name)
        if thread_id < 0 :
            thread_id = max_int32 + thread_id

        print("{} {} {} {} {} {} {} {} {} {} {} {} {} ".format(layer_name, layer_type, phase, marker_start - time_base, marker_end - time_base, marker_end - marker_start, start_time - time_base, end_time - time_base, end_time - start_time, corr_id, thread_id, ker_name, pivot_tbl_tag), file=file_des)
        report_tbl['LayerName'].append(layer_name)
        report_tbl['LayerType'].append(layer_type)
        report_tbl['Phase'].append(phase)
        report_tbl['CPUStartTime(ns)'].append(marker_start - time_base)
        report_tbl['CPUEndTime(ns)'].append(marker_end - time_base)
        report_tbl['CPUDuration(ns)'].append(marker_end - marker_start)
        report_tbl['GPUStartTime(ns)'].append(start_time - time_base)
        report_tbl['GPUEndTime(ns)'].append(end_time - time_base)
        report_tbl['GPUDuration(ns)'].append(end_time - start_time)
        report_tbl['CorrId'].append(corr_id)
        report_tbl['Thread'].append(thread_id)
        report_tbl['Kernel'].append(ker_name)
        report_tbl['ExperTag'].append(pivot_tbl_tag)

    ## Create Pandas data frame
    data_frame = pd.DataFrame(report_tbl)
    del report_tbl

    return data_frame

################################################################################
## get_layer_type_from_name()
##
################################################################################
def get_layer_type_from_name(name=None) :
   """
   get the layer type from the long form layer name
   """
   layer_type = name
   if name is None :
       print("Error get_layer_type_from_name - Bad args - exiting...")
       sys.exit(1)

   pattern = re.compile(r"layer_\d+_\d+_(\w+)")
   res = re.match(pattern, name)
   if res is not None:
       layer_type = "{}".format(res.group(1))
   layer_type = re.sub(r"\d+", "", layer_type)
   return layer_type
################################################################################
## demangle_kernel_name()
##
################################################################################
def demangle_kernel_name(name=None) :
    """
    Kernel names are mangled and I can't find a good linux tool to demangle the cuda names
    """
    demangle = re.compile(r"_Z[A-Z0-9]\d+")
    new_name = re.sub(demangle, "", name)
    demangle = re.compile(r"(\S+)Li\d+")
    res = re.match(demangle, new_name)
    if res is not None :
        new_name = res.group(1)
    res = re.match(r"([a-zA-Z]+)\d+([a-zA-Z]+)\d+(\w+)", new_name)
    if res is not None :
        new_name = "{}::{}::{}".format(res.group(1), res.group(2), res.group(3))

    return new_name
################################################################################
## get_tbl_name_from_type()
##
################################################################################
def get_tbl_name_from_type(tbl_type=None, tbl_list=None) :
    """
    Return full table name that matches the tbl_type substring
    """
    tbl_name = None
    if tbl_type is None:
        print ("Error get_tbl_name_from_pattern: No tbl_type specified - exiting.\n")
        sys.exit(1)

    if tbl_list is None:
        print ("Error get_tbl_name_from_pattern: No tbl_list specified - exiting.\n")
        sys.exit(1)


    ## Walk the list of tbls - return the one that has substring tbl_type
    for tbl in tbl_list :
        pattern = re.compile(tbl_type)
        if pattern.search(tbl) :
            tbl_name = tbl
            break

    return tbl_name

################################################################################
## get_tbl_marker_by_time_window()
##
##  @@@ This function is slow - sql look ups seem to take really long
##  Try using Pandas instead - create a frame
##  Search by time stamp
##  return the fields
##
################################################################################
def get_tbl_marker_by_time_window(cpu_start=None, cpu_end=None, pd_frame=None) :
    """
        Find the marker / range whose start and end times cover the cpu event
        start and end times passed in
    """
    if cpu_start is None or cpu_end is None or pd_frame is None:
        print ("get_tbl_marker_by_time_window: Bad args - exiting ")
        sys.exit(1)

    ## Get the first entry whose time stamp is > end
    ## record the ID - then do a 2nd query that returns name when ID == id from prev query
    query_string     = "timestamp > {}".format(cpu_end)
    tmp_frame        = pd_frame.query(query_string)
    pd_marker_id     = tmp_frame['id'].values.tolist()[0]
    marker_end       = tmp_frame['timestamp'].values.tolist()[0]

    query_string     = "id == {}".format(pd_marker_id)
    tmp_frame        = pd_frame.query(query_string)
    pd_name_id       = tmp_frame['name_id'].values.tolist()[0]
    marker_name      = tmp_frame['name'].values.tolist()[0]
    marker_start     = tmp_frame['timestamp'].values.tolist()[0]

    if(Debug) :
        print ("Marker name {} start {} end {} ".format(marker_name, marker_start, marker_end))
    return marker_name, marker_start, marker_end

################################################################################
## get_tbl_event_by_corr_id()
##
################################################################################
def get_tbl_event_by_corr_id(corr_id=None, pd_frame=None) :
    if corr_id is None or pd_frame is None:
        print ("Error get_runtime_event_by_corr_id: missing argument - exiting.\n")
        sys.exit(1)
    ## use panda frame instead of sql query
    query_string = "correlationId == {}".format(corr_id)
    tmp_frame = pd_frame.query(query_string)

    start     = tmp_frame['start'].values.tolist()[0]
    end       = tmp_frame['end'].values.tolist()[0]
    thread_id = tmp_frame['threadId'].values.tolist()[0]

    return [start, end, thread_id]
################################################################################
## name_lookup_by_id()
##
################################################################################
def tbl_name_lookup_by_id(cur=None, name_id=None) :
    if name_id is None:
        print ("Error name_lookup_by_id - no name specified - exiting...")
        sys.exit(1)

    if cur is None:
        print ("Error process_runtime_tbl: No cursor specified - exiting.\n")
        sys.exit(1)

    query_string = "select value from StringTable where _id_={0}".format(name_id)

    return

################################################################################
## process_runtime_tbl()
##
##   Decode the RUNTIME table
################################################################################
def process_runtime_tbl(tbl=None, cur=None):
    if tbl is None:
        print ("Error process_runtime_tbl: No tbl specified - exiting.\n")
        sys.exit(1)
    if cur is None:
        print ("Error process_runtime_tbl: No cursor specified - exiting.\n")
        sys.exit(1)

    pattern = re.compile('RUNTIME')
    if pattern.search(tbl) :
        cmd_string = "select * from {};".format(tbl)
        print ("Executing sql cmd {}".format(cmd_string), end='\n')
        cur.execute(cmd_string)   ## Need to use a tuple for variable sub- even though only passing 1 value
        tbl_hdr = get_tbl_hdrs(cur, Debug)
        dump_rows_runtime(cur, tbl_hdr)


    return
################################################################################
## Main program
################################################################################

## Call the functions
parse_cmd_line()
#run_io_examples()
output_fd = open_ouput_file()

frame_list = []
for db_file in db_file_list :
    pd_frame = read_db_file(db_file, output_fd);
    frame_list.append(pd_frame)
    panda_sheet  = db_file
    pd_frame.to_excel(excel_writer, panda_sheet)

panda_sheet = 'combined_tbl'
pivot_tbl_frame = pd.concat(frame_list)
## Combine all the frames into 1
pivot_tbl_frame.to_excel(excel_writer, panda_sheet)

## Close the xcel sheet
excel_writer.save()
if pivot_tbl is not None:
    output_fd.close()
