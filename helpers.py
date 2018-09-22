from datetime import datetime, timedelta
import time

def curr_time(offset=7):
    return datetime.now() - timedelta(hours=offset) # offset from UTC to PST

def make_header(s):
    return ("#" * 42) + ("\n{:^42}\n".format(s)) + ("#" * 42)
    
def print_and_log(logfile, s):
    if logfile:
        with open(logfile, 'a') as log:
            log.write(str(s))
            log.write("\n")
    print(s)
        
def print_and_log_header(logfile, s):
    h = make_header(str(s))
    if logfile:
        with open(logfile, 'a') as log:
            log.write(h)
            log.write("\n")
    print(h)
    
def progress_bar(curr, total, tabs=0, prefix="", logfile=None):
    block_char = u"\u2588"
    pct_done = curr / total * 100
    blocks = int(pct_done // 2)
    blocks *= block_char
    additional = [u"\u258F", u"\u258E", u"\u258D", u"\u258C", u"\u258B", u"\u258A", u"\u2589"]
    i = int((pct_done % 2) * 4) - 1
    if i >= 0:
        blocks += additional[i]
    tabs *= "\t"
    ret = "\r" if curr != 0 else ""
    end = "" if pct_done < 100 else "\n"
    s = "{}{}{}|{:<50}| {:.2f}% ({}/{})".format(ret, tabs, prefix, blocks, pct_done, curr, total)
    if logfile:
        with open(logfile, 'a') as log:
            log.write(s.strip() + "\n")
    print(s, end=end)
    
def sec_to_str(secs):
    ms = secs - int(secs)
    days = int(secs // (24 * 3600))
    hours = int((secs % ((24 * 3600))) // 3600)
    minutes = int((secs % 3600) // 60)
    seconds = int(secs % 60)
    return "{:02}:{:02}:{:02}:{:02}.{}".format(days, hours, minutes, seconds, "{:.3}".format(ms)[2:])

def timer(f, *args, log=None):
    print_and_log(log, "Start: {}".format(curr_time()))
    start = time.time()
    result = f(*args)
    end = time.time()
    print_and_log(log, "End: {}".format(curr_time()))
    print_and_log(log, "Finished in {}".format(sec_to_str(end - start)))
    return result