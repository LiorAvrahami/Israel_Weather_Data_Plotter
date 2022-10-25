import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as plt_animation
from matplotlib.colors import ListedColormap, Normalize
import os
import cycler
import itertools
from typing import List,Tuple,Optional,Dict,Literal,Union,Iterable

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

from load_data import get_days_of_years,years_times,fix_hebrew_for_print,years_data,years_titles,get_time_from_day,get_day_of_year,years_data_accuracy,data_confines,get_accuracy_of_data_index,get_location_title_of_data
from time_span_defs import *

def day_to_text(time_in_days):
    years = get_time_from_day(time_in_days, "Y")
    time_in_days -= get_day_of_year(years)
    months = get_time_from_day(time_in_days, "M")
    time_in_days -= get_day_of_year(months)
    days = get_time_from_day(time_in_days, "D")
    time_in_days -= get_day_of_year(days)
    hours = get_time_from_day(time_in_days, "h")
    time_in_days -= get_day_of_year(hours)
    minuets = get_time_from_day(time_in_days, "m")
    time_in_days -= get_day_of_year(minuets)

    years = years.astype(int)
    months = months.astype(int)
    days = days.astype(int)
    hours = hours.astype(int)
    minuets = minuets.astype(int)

    ret = ""
    if years != 0:
        ret += "{}Y {}M {}D".format(years,months,days)
    elif months != 0:
        ret += "{}M {}D".format(months, days)
    elif days != 0:
        if days != 1:
            ret += "{}D".format(days)
        else:
            hours = 24
    if minuets != 0 or hours != 0 or ret == "":
        ret += " {}:{}".format(hours,minuets)
    if ret[0] == " ":
        ret = ret[1:]
    return ret

def time_segment_to_text(segment):
    return "({},{})".format(day_to_text(segment[0]),day_to_text(segment[1]))

get_minimal_num_of_bins_from_accuracy = lambda accuracy,hist_dat:(np.nanmax(hist_dat)-np.nanmin(hist_dat))/(2*accuracy)

def get_bins(accuracy_x,hist_dat_x,accuracy_y,hist_dat_y):
    num_of_bins = int(min(len(hist_dat_x) ** 0.5, get_minimal_num_of_bins_from_accuracy(accuracy_x, hist_dat_x),
                       get_minimal_num_of_bins_from_accuracy(accuracy_y, hist_dat_y)))
    bins = (np.linspace(min(hist_dat_x),max(hist_dat_x),num_of_bins),np.linspace(min(hist_dat_y),max(hist_dat_y),num_of_bins))
    return bins

def get_histogram_x_y_data(parts_of_day,parts_of_year,q1,q2):
    hist_dat_x_array = []
    hist_dat_y_array = []
    for year in years_data.keys():
        days = get_days_of_years(years_times[year])
        indexes_time_of_day = is_in_segments((days % 1), parts_of_day)

        indexes_part_of_year = is_in_segments(days, parts_of_year)

        indexes = indexes_time_of_day * indexes_part_of_year
        indexes *= np.isfinite(years_data[year][:, q1].astype(np.float)) * np.isfinite(years_data[year][:, q2].astype(np.float))
        hist_dat_x_array.append(years_data[year][indexes, q1].astype(np.float))
        hist_dat_y_array.append(years_data[year][indexes, q2].astype(np.float))
    hist_dat_x, hist_dat_y = np.concatenate(hist_dat_x_array), np.concatenate(hist_dat_y_array)
    return hist_dat_x, hist_dat_y

def color_to_cmap(r,g,b):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(r, r/5, N)
    vals[:, 1] = np.linspace(g, g/5, N)
    vals[:, 2] = np.linspace(b, b/5, N)
    vals[:, 3] = np.linspace(0, 1, N)
    return ListedColormap(vals)

def calculate_hist(q1, q2, target_parts_of_day, parts_of_year, histedges = None, b_normalize_column = False, b_logscale=True):
    if parts_of_year == None:
        parts_of_year = ALLYEAR
    if target_parts_of_day == None:
        target_parts_of_day = ALLDAY

    hist_dat_x, hist_dat_y = get_histogram_x_y_data(target_parts_of_day,parts_of_year,q1,q2)

    if histedges == None:
        bins = get_bins(get_accuracy_of_data_index(q1),hist_dat_x,get_accuracy_of_data_index(q2),hist_dat_y)
    else:
        bins = histedges
    H, xedges, yedges = np.histogram2d(hist_dat_x, hist_dat_y, bins)
    if b_logscale:
        H = np.log(H)
    if b_normalize_column:
        H_sum = H.sum()
        H = H / H.max(axis=0)
        H = H * H_sum / np.nansum(H)
    return {"hist":H, "xedges":xedges, "yedges":yedges}

def draw_hist(q1, q2, hist, xedges, yedges, target_parts_of_day, parts_of_year, norm_max_val=None, axes=None, cmap=None, data_name=None, highlight_edges=False):
    sample_year = list(years_data.keys())[0]
    if type(cmap) == tuple or type(cmap) == list or type(cmap) == np.ndarray:
        cmap = color_to_cmap(*cmap)
    if data_name == None:
        data_name = ""

        if parts_of_year == SUMMER:
            data_name += "Summer"
        elif parts_of_year == WINTER:
            data_name += "Winter"
        elif parts_of_year == TRANSITION:
            data_name = "Transition seasons"
        elif parts_of_year == ALLYEAR:
            data_name += "All Year"
        else:
            data_name += "costume part of year"
        data_name += ", "
        if target_parts_of_day == DAY:
            data_name += "Day time"
        elif target_parts_of_day == NIGHT:
            data_name += "Night time"
        elif target_parts_of_day == ALLDAY:
            data_name += "All Day"
        else:
            data_name += "costume part of day"
    if axes == None:
        _,axes = plt.subplots()
    axes.set_xlabel(fix_hebrew_for_print(years_titles[sample_year][q1]))
    axes.set_ylabel(fix_hebrew_for_print(years_titles[sample_year][q2]))
    axes.set_xlim(*data_confines[q1])
    axes.set_ylim(*data_confines[q2])
    H = hist.T
    Hf = H.flatten()
    if norm_max_val == None:
        norm = Normalize().autoscale(A=H)
    else:
        norm = Normalize(vmin=0, vmax=norm_max_val)

    if highlight_edges:
        if type(cmap) != str and cmap != None:
            edge_colors = cmap(Hf.astype(float) / np.max(Hf))
            edge_colors[:, :3] /= 2.1
            # edge_colors[:, 3] = np.minimum(edge_colors[:, 3], 0.07)
        else:
            edge_colors = "k"
    else:
        edge_colors = None
    image = axes.pcolormesh(xedges, yedges, H, cmap=cmap, edgecolors=edge_colors, linewidth=0.1, norm=norm)
    if data_name != "":
        cbar = plt.colorbar(image, ticks=[], aspect=100, fraction=0.05, pad=0)
        cbar.ax.set_ylabel(data_name)

def plot_hist(q1, q2, target_parts_of_day = None, histedges = None,norm_max_val=None, parts_of_year = None, axes = None, cmap=None, data_name=None, b_normalize_column = False, b_logscale=True,highlight_edges = False):
    ret = calculate_hist(q1 = q1, q2 = q2, target_parts_of_day = target_parts_of_day, parts_of_year = parts_of_year, histedges = histedges, b_normalize_column = b_normalize_column, b_logscale = b_logscale)
    draw_hist(q1, q2, **ret, target_parts_of_day=target_parts_of_day, norm_max_val=norm_max_val, parts_of_year=parts_of_year, axes=axes, cmap=cmap, data_name=data_name, highlight_edges=highlight_edges)
    return ret

def year_hist(q1, q2, target_parts_of_day = None, b_normalize_column=False, b_logscale=False):
    _,axes = plt.subplots()
    if target_parts_of_day == None:
        target_parts_of_day = ALLDAY

    if target_parts_of_day == DAY:
        title = "Day time: "
    elif target_parts_of_day == NIGHT:
        title = "Night time: "
    elif target_parts_of_day == ALLDAY:
        title = "All Day: "
    else:
        title = "costume part of day: "
    segments_txt = []
    for seg_of_day in sorted(target_parts_of_day):
        segments_txt.append("{}".format(time_segment_to_text(seg_of_day)))
    title = "seasonal comparison. reduced to "+ title + ", ".join(segments_txt)

    axes.set_title(title)
    plot_hist(q1, q2, target_parts_of_day=target_parts_of_day, b_normalize_column=b_normalize_column, b_logscale=b_logscale, parts_of_year=SUMMER, axes=axes,cmap=(0.9, 0.2, 0.2))
    plot_hist(q1, q2, target_parts_of_day=target_parts_of_day, b_normalize_column=b_normalize_column, b_logscale=b_logscale, parts_of_year=TRANSITION, axes=axes, cmap=(0.2, 0.9, 0.2))
    plot_hist(q1, q2, target_parts_of_day=target_parts_of_day, b_normalize_column=b_normalize_column, b_logscale=b_logscale, parts_of_year=WINTER, axes=axes, cmap=(0.2, 0.2, 0.9))

def compare_day_night(q1, q2, parts_of_year = None,b_normalize_column=False, b_logscale=False):
    _, axes = plt.subplots()
    if parts_of_year == None:
        parts_of_year = ALLYEAR

    if parts_of_year == SUMMER:
        title = "Summer time: "
    elif parts_of_year == WINTER:
        title = "Winter time: "
    elif parts_of_year == TRANSITION:
        title = "Transition seasons: "
    elif parts_of_year == ALLYEAR:
        title = "All Year: "
    else:
        title = "costume part of year: "
    segments_txt = []
    for seg_of_day in sorted(parts_of_year):
        segments_txt.append("{}".format(time_segment_to_text(seg_of_day)))
    title = "Day-Night comparison. reduced to "+ title + ", ".join(segments_txt)

    axes.set_title(title)
    plot_hist(q1, q2, target_parts_of_day=DAY, b_normalize_column=b_normalize_column, b_logscale=b_logscale, parts_of_year=parts_of_year, cmap=(0.9, 0.2, 0.2), axes=axes)
    plot_hist(q1, q2, target_parts_of_day=NIGHT, b_normalize_column=b_normalize_column, b_logscale=b_logscale, parts_of_year=parts_of_year, cmap=(0.05, 0.15, 0.7), axes=axes)


# def creat_day_night_cycle_animation(q1, q2,time_points_in_moving_average=None, parts_of_year = None,colormaps = None,b_normalize_column=None, b_logscale=None,highlight_edges=None):
#     # Defults:
#     colormaps = [(0.9, 0.2, 0.2), (0.2, 0.2, 0.9), (0.2, 0.9, 0.2)] if colormaps == None else colormaps
#     b_normalize_column = False if b_normalize_column == None else b_normalize_column
#     b_logscale = False if b_logscale == None else b_logscale
#     time_points_in_moving_average = 6 if time_points_in_moving_average == None else time_points_in_moving_average
#     highlight_edges = True
#
#     times_of_day = split_day(time_points_in_moving_average)
#
#     # Init visuals
#     fig, axes = plt.subplots()
#     histedges = {}
#     for i in range(len(parts_of_year)):
#         part_of_year, cmap = parts_of_year[i], colormaps[i]
#         ret =  plot_hist(q1, q2, target_parts_of_day=ALLDAY, b_normalize_column=b_normalize_column, b_logscale=b_logscale,parts_of_year=part_of_year, cmap=cmap, axes=axes,data_name="",highlight_edges=highlight_edges)
#         histedges[i] = (ret["xedges"],ret["yedges"])
#
#     b_first = [None]
#     def update(frame,num_of_frames = None):
#         if num_of_frames != None:
#             print("frame: {} \t of: {}".format(frame,num_of_frames))
#         axes.clear()
#         for i in range(len(parts_of_year)):
#             part_of_year, cmap = parts_of_year[i], colormaps[i]
#             plot_hist(q1, q2, target_parts_of_day=[times_of_day[frame % len(times_of_day)]], histedges=histedges[i], parts_of_year=part_of_year, cmap=cmap, axes=axes, data_name=b_first[0], b_normalize_column=b_normalize_column, b_logscale=b_logscale,highlight_edges=highlight_edges)
#         axes.set_title("Time Of Day: {}:00Z".format(int(sum(times_of_day[frame%len(times_of_day)])/2*24)))
#         b_first[0] = ""
#
#     num_of_frames_till_loop = len(times_of_day)
#     return axes, fig, update,num_of_frames_till_loop
#
# def animate_day_night_cycle_on_screan(q1, q2,frame_time_ms=1, parts_of_year = None,colormaps = None,b_normalize_column=None, b_logscale=None):
#     axes, fig, update, num_of_frames_till_loop = creat_day_night_cycle_animation(q1=q1, q2=q2, parts_of_year=parts_of_year, colormaps=colormaps, b_normalize_column=b_normalize_column, b_logscale=b_logscale)
#     a = plt_animation.FuncAnimation(fig, update, interval=frame_time_ms)
#     plt.show()
#
# def save_day_night_cycle_animation(q1, q2, fps=15, file_name=r".\animation\day_night.mp4",time_points_in_moving_average=6,num_of_loops=1, parts_of_year=None, colormaps=None, b_normalize_column=None, b_logscale=None):
#     FFMpegWriter = plt_animation.writers['ffmpeg']
#     metadata = dict(title='Day Night Weather Comparison Animation', artist='Lior Avrahami - Matplotlib',
#                     comment='Movie support!')
#     writer = FFMpegWriter(fps=60, metadata=metadata)
#     axes, fig, update, num_of_frames_till_loop = creat_day_night_cycle_animation(q1=q1, q2=q2,time_points_in_moving_average=time_points_in_moving_average, parts_of_year=parts_of_year, colormaps=colormaps, b_normalize_column=b_normalize_column, b_logscale=b_logscale)
#     num_of_frames = (num_of_frames_till_loop) * num_of_loops
#     a = plt_animation.FuncAnimation(fig, update, fargs=[num_of_frames], save_count=num_of_frames)
#     handle_overite(file_name)
#     a.save(file_name,writer=writer,dpi=500)
#     del a

def split_day(days_in_moving_average,total_number_of_frames):
    parts_of_day = np.linspace(0,1,total_number_of_frames,endpoint=False)
    lower_bounds = parts_of_day - days_in_moving_average / 2
    upper_bounds = parts_of_day + days_in_moving_average / 2
    return list(zip(lower_bounds, upper_bounds))

def split_year(days_in_moving_average,total_number_of_frames):
    days_of_year = np.linspace(0,365,total_number_of_frames,endpoint=False)
    lower_bounds = days_of_year - days_in_moving_average / 2
    upper_bounds = days_of_year + days_in_moving_average / 2
    return list(zip(lower_bounds, upper_bounds))

# def creat_season_cycle_animation(q1, q2,time_points_in_moving_average=None, target_parts_of_day = None,colormaps = None,b_normalize_column=None, b_logscale=None):
#     # Defults:
#     colormaps = [(0.9, 0.2, 0.2), (0.2, 0.2, 0.9), (0.2, 0.9, 0.2)] if colormaps == None else colormaps
#     b_normalize_column = False if b_normalize_column == None else b_normalize_column
#     b_logscale = False if b_logscale == None else b_logscale
#     time_points_in_moving_average = 6 if time_points_in_moving_average == None else time_points_in_moving_average
#
#     fig, axes = plt.subplots()
#     times_of_year = split_year(time_points_in_moving_average)
#     histedges = {}
#     for i in range(len(target_parts_of_day)):
#         part_of_day, cmap = target_parts_of_day[i], colormaps[i]
#         ret =  plot_hist(q1, q2, target_parts_of_day=part_of_day, b_normalize_column=b_normalize_column, b_logscale=b_logscale,parts_of_year=ALLYEAR, cmap=cmap, axes=axes,data_name="")
#         histedges[i] = (ret["xedges"],ret["yedges"])
#     b_first = [None]
#     def update(frame,num_of_frames = None):
#         if num_of_frames != None:
#             print("frame: {} \t of: {}".format(frame,num_of_frames))
#         axes.clear()
#         for i in range(len(target_parts_of_day)):
#             target_part_of_day, cmap = target_parts_of_day[i], colormaps[i]
#             plot_hist(q1, q2, target_parts_of_day=target_part_of_day, histedges=histedges[i], parts_of_year=[times_of_year[frame % len(times_of_year)]], cmap=cmap, axes=axes, data_name=b_first[0], b_normalize_column=b_normalize_column, b_logscale=b_logscale)
#         day_of_year_to_print = int(sum(times_of_year[frame%len(times_of_year)])/2)
#         Season_name = " "
#         if is_in_segments(day_of_year_to_print,SUMMER):
#             Season_name = "Summer"
#         if is_in_segments(day_of_year_to_print,WINTER):
#             Season_name = "Winter"
#         if is_in_segments(day_of_year_to_print,TRANSITION):
#             Season_name = "Transition"
#         axes.set_title("Day of year: {}:00Z\n Season: {}".format(day_of_year_to_print,Season_name))
#         b_first[0] = ""
#     num_of_frames_till_loop = len(times_of_year)
#     return axes, fig, update,num_of_frames_till_loop
#
# def animate_season_cycle_on_screan(q1, q2,frame_time_ms=1, target_parts_of_day = None,colormaps = None,b_normalize_column=None, b_logscale=None):
#     axes, fig, update, num_of_frames_till_loop = creat_season_cycle_animation(q1=q1, q2=q2, target_parts_of_day=target_parts_of_day, colormaps=colormaps, b_normalize_column=b_normalize_column, b_logscale=b_logscale)
#     a = plt_animation.FuncAnimation(fig, update, interval=frame_time_ms)
#     plt.show()
#
# def save_season_cycle_animation(q1, q2, fps=15, file_name=r".\animation\seasons.mp4",time_points_in_moving_average=12,num_of_loops=1, target_parts_of_day=None, colormaps=None, b_normalize_column=None, b_logscale=None):
#     FFMpegWriter = plt_animation.writers['ffmpeg']
#     metadata = dict(title='Year Long Weather Comparison Animation', artist='Lior Avrahami - Matplotlib',
#                     comment='Movie support!')
#     writer = FFMpegWriter(fps=60, metadata=metadata)
#     axes, fig, update, num_of_frames_till_loop = creat_season_cycle_animation(q1=q1, q2=q2,time_points_in_moving_average=time_points_in_moving_average, target_parts_of_day=target_parts_of_day, colormaps=colormaps, b_normalize_column=b_normalize_column, b_logscale=b_logscale)
#     num_of_frames = (num_of_frames_till_loop) * num_of_loops
#     a = plt_animation.FuncAnimation(fig, update,fargs=[num_of_frames], save_count=num_of_frames)
#     handle_overite(file_name)
#     a.save(file_name,writer=writer,dpi=500)
#     del a

def creat_day_night_animation(q1, q2,days_in_moving_average=None,total_number_of_frames=None, parts_of_year:Optional[List[PartOfYear]] = None,colormaps = None,highlight_edges=None):
    days_in_moving_average = 6 if days_in_moving_average == None else days_in_moving_average
    times_of_day = split_day(days_in_moving_average,total_number_of_frames)
    times_of_day = [[t] for t in times_of_day]
    time_cyclers_arr = []
    for part_of_year in parts_of_year:
        times_of_year = (part_of_year,)*len(times_of_day)
        time_cyclers_arr.append(cycler.cycler(day=times_of_day,year=times_of_year))
    return creat_cycle_animation(q1,q2,time_cyclers_arr,colormaps)

def creat_seasons_animation(q1, q2,days_in_moving_average=None,total_number_of_frames=None, parts_of_day:Optional[List[PartOfDay]] = None,colormaps = None,highlight_edges=None):
    days_in_moving_average = 6 if days_in_moving_average == None else days_in_moving_average
    days_of_year = split_year(days_in_moving_average,total_number_of_frames)
    days_of_year = [[t] for t in days_of_year]
    time_cyclers_arr = []
    for part_of_day in parts_of_day:
        part_of_day_arr = (part_of_day,)*len(days_of_year)
        time_cyclers_arr.append(cycler.cycler(day=part_of_day_arr,year=days_of_year))
    return creat_cycle_animation(q1,q2,time_cyclers_arr,colormaps,highlight_edges=highlight_edges)

def creat_cycle_animation(q1, q2,time_cyclers_arr,colormaps = None,highlight_edges=None):
    """
    time_cyclers_iter:  array of cyclers, each element of wich is a cycler with "day", "year" keys. these are to be cycled during the animation
    """
    # Defults:
    colormaps = [(1, 0, 0), (0, 0, 1), (0, 1, 0)] if colormaps == None else colormaps
    highlight_edges = True if highlight_edges == None else highlight_edges

    TimeFilter = Dict[Literal["day","year"],Union[PartOfDay,PartOfYear]]
    histograms_data:List[Tuple[Iterable[np.ndarray,TimeFilter,Tuple],float]] = [None] * len(time_cyclers_arr)
    # Calculate histograms for each time splitting
    for hist_channle_index,cyc in enumerate(time_cyclers_arr):
        merged_filter = merge_time_filter_iterable(cyc)
        hist_dat_x, hist_dat_y = get_histogram_x_y_data(merged_filter["day"],merged_filter["year"],q1,q2)
        bins = get_bins(get_accuracy_of_data_index(q1), hist_dat_x, get_accuracy_of_data_index(q2), hist_dat_y)

        hist_data_cycle = []
        for time_filter in cyc:
            time_filter:TimeFilter
            ret = calculate_hist(q1, q2, time_filter["day"], time_filter["year"],histedges=bins)
            ret.update({"target_parts_of_day": time_filter["day"], "parts_of_year": time_filter["year"]})
            hist_data_cycle.append(ret)
        max_hist_val = max([np.max(histogram["hist"]) for histogram in hist_data_cycle])
        histograms_data[hist_channle_index] = (itertools.cycle(hist_data_cycle),max_hist_val)

    fig = plt.figure()
    axes = fig.subplots()
    b_first = [None]
    def update(frame,num_of_frames = None):
        if num_of_frames != None:
            print("frame: {} \t of: {}".format(frame,num_of_frames))
        axes.clear()
        filters = []
        for i,hist_data in enumerate(histograms_data):
            hist_data_cycle, max_hist_val = hist_data
            cur_hist_data = next(hist_data_cycle)
            draw_hist(q1,q2,**cur_hist_data,norm_max_val=max_hist_val,axes=axes,cmap=colormaps[i],data_name=b_first[0],highlight_edges=highlight_edges)
            filters.append({"day":cur_hist_data["target_parts_of_day"], "year":cur_hist_data["parts_of_year"]})
        title_lines = ["whether distrebution in \"{}\" vs time".format(get_location_title_of_data())]
        common_filter = get_common_filter(filters)
        day_of_year_to_print,hour_of_day_to_print = filter_segments_to_points(common_filter)
        if "day" in common_filter and hour_of_day_to_print is not None:
            title_lines.append("Time Of Day: {}:00Z".format(hour_of_day_to_print))
        if "year" in common_filter and day_of_year_to_print is not None:
            Season_name = " "
            if is_in_segments(day_of_year_to_print, SUMMER):
                Season_name = "Summer"
            if is_in_segments(day_of_year_to_print, WINTER):
                Season_name = "Winter"
            if is_in_segments(day_of_year_to_print, TRANSITION):
                Season_name = "Transition"
            title_lines.append("Day of year: {}\n Season: {}".format(day_of_year_to_print, Season_name))
        axes.set_title("\n".join(title_lines))
        b_first[0] = ""

    num_of_frames_till_loop = len(time_cyclers_arr[0])
    return axes, fig, update,num_of_frames_till_loop

def handle_overite(file_name):
    if os.path.isfile(file_name):
        old_file_name = file_name
        while os.path.isfile(file_name):
            split_name = os.path.splitext(file_name)
            file_name = split_name[0] + " old" + split_name[1]
        os.rename(old_file_name,file_name)

def save_cycle_animation(cycle_animation_ret, fps, file_name, num_of_loops):
    FFMpegWriter = plt_animation.writers['ffmpeg']
    metadata = dict(title='Year Long Weather Comparison Animation', artist='Lior Avrahami - Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    axes, fig, update, num_of_frames_till_loop = cycle_animation_ret
    num_of_frames = (num_of_frames_till_loop) * num_of_loops
    a = plt_animation.FuncAnimation(fig, update, fargs=[num_of_frames], save_count=num_of_frames)
    handle_overite(file_name)
    a.save(file_name, writer=writer, dpi=300)
    del a

def animate_season_cycle_on_screan(q1, q2,frame_time_ms=1, parts_of_day = None,colormaps = None):
    axes, fig, update, num_of_frames_till_loop = creat_seasons_animation(q1=q1, q2=q2, parts_of_day=parts_of_day, colormaps=colormaps)
    a = plt_animation.FuncAnimation(fig, update, interval=frame_time_ms)
    plt.show()

def save_season_cycle_animation(q1, q2, fps=40,total_number_seconds=3, file_name=r".\animation\seasons.mp4",days_in_moving_average=12,num_of_loops=1, parts_of_day=None, colormaps=None,highlight_edges=False):
    seasons_cycle_animation_ret = creat_seasons_animation(q1=q1, q2=q2, days_in_moving_average=days_in_moving_average, total_number_of_frames=int(fps*total_number_seconds),
                            parts_of_day=parts_of_day, colormaps=colormaps, highlight_edges=highlight_edges)
    save_cycle_animation(seasons_cycle_animation_ret,fps,file_name,num_of_loops)

def save_day_night_cycle_animation(q1, q2, fps=40,total_number_seconds=3, file_name=r".\animation\day night.mp4",days_in_moving_average=0.04,num_of_loops=1, parts_of_year=None, colormaps=None,highlight_edges=False):
    seasons_cycle_animation_ret = creat_day_night_animation(q1=q1, q2=q2, days_in_moving_average=days_in_moving_average, total_number_of_frames=int(fps*total_number_seconds),
                            parts_of_year=parts_of_year, colormaps=colormaps, highlight_edges=highlight_edges)
    save_cycle_animation(seasons_cycle_animation_ret,fps,file_name,num_of_loops)

if __name__ == "__main__":
    # animate_day_night_cycle_on_screan(1,2,parts_of_year=[WINTER,SUMMER])
    save_day_night_cycle_animation(1,2,parts_of_year=[SUMMER,WINTER],)
    pass
# plot_hist(1,2,b_logscale=False,cmap=(0.9, 0.2, 0.2))
# plt.show()

# save_day_night_cycle_animation(4,2,parts_of_year=[SUMMER,WINTER],num_of_loops=3)
# save_season_cycle_animation(4,2,target_parts_of_day=[DAY,NIGHT])
# save_season_cycle_animation(1,2,target_parts_of_day=[DAY,NIGHT]))
