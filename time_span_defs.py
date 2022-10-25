ALLYEAR = [(0,400)]
SUMMER = [(180,270)]
WINTER = [(350,400),(0,50)]
TRANSITION = [(270,350),(50,180)]

DAY = [(0.35,0.7)]
NIGHT = [(0.9,1),(0,0.2)]
ALLDAY = [(0,1)]

def is_in_segments(test_nums,segments):
    Truth_vals = (test_nums == 0)*False
    for segment in segments:
        Truth_vals += (test_nums >= segment[0]) * (test_nums <= segment[1])
    return Truth_vals

def merge_time_filter_iterable(filter_itr):
    all_allowed_time_of_day = [filt["day"] for filt in filter_itr]
    all_allowed_time_of_year = [filt["year"] for filt in filter_itr]
    from itertools import chain
    return {"day":list(chain(*all_allowed_time_of_day)),
            "year": list(chain(*all_allowed_time_of_year))}

def get_common_filter(filter_itr):
    b_daytime_is_same = True
    daytime = None
    b_yeartime_is_same = True
    yeartime = None
    for filter in filter_itr:
        if daytime is None:
            daytime = filter["day"]
        else:
            if filter["day"] != daytime:
                b_daytime_is_same = False
        if yeartime is None:
            yeartime = filter["year"]
        else:
            if filter["year"] != yeartime:
                b_yeartime_is_same = False
    if b_yeartime_is_same and b_daytime_is_same:
        return {"day":daytime,"year":yeartime}
    if (not b_yeartime_is_same) and b_daytime_is_same:
        return {"day":daytime}
    if b_yeartime_is_same and (not b_daytime_is_same):
        return {"year":yeartime}
    if (not b_yeartime_is_same) and (not b_daytime_is_same):
        return {}

def filter_segments_to_points(filter):
    time_of_year,time_of_day = None,None
    if "day" in filter:
        if len(filter["day"]) == 1:
            time_of_day = int(sum(filter["day"][0]) / 2 * 24)
    if "year" in filter:
        if len(filter["year"]) == 1:
            time_of_year = int(sum(filter["year"][0]) / 2)
    return time_of_year, time_of_day

from typing import List as _List,Tuple as _Tuple,Optional as _Optional
PartOfYear = _List[_Tuple[int,int]]
PartOfDay = _List[_Tuple[float,float]]