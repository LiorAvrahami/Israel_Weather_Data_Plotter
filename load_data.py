import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as plt_animation
from matplotlib.colors import ListedColormap, Normalize
import os
import re
from cycler import cycler

# workdir = 'C:\Users\Lior\Downloads\Chrome\israel_weather_history'

def reverse_date(date_str:str):
    return "-".join(reversed(date_str.split("-")))

def fix_hebrew_for_print(txt):
    # flip txt, than flip all the non hebrew words in txt
    heb_alphabet = "פםןוטארקךלחיעכגדשתצמנהבסז"
    txt = txt[::-1] #flip
    words = re.split(r"[ ()]", txt)
    for i in range(len(words)):
        b_is_all_word_hebrew = np.all([letter in heb_alphabet for letter in np.unique(list(words[i]))])
        if not b_is_all_word_hebrew:
            words[i] = words[i][::-1]
    return " ".join(words)


data_confines = None
loaded_years = {}
years_data = {}
years_data_accuracy = {}
years_times = {}
years_titles = {}

def print_data_titles():
    titles = list(years_titles.values())[0]
    for r in range(len(titles)):
        print("{}: ".format(r))
        print(titles[r])
PT = print_data_titles


data_files = os.listdir('.\israel_weather_history')
for data_file in data_files:
    with open(os.path.join('.\israel_weather_history',data_file), 'r',encoding='Windows-1255') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array(list(reader))
        titles = data[0]
        data = data[1:,:].astype(object)
        data[:, 3:][data[:, 3:] == '-'] = np.nan
        data[:,3:] = data[:,3:].astype(float)

        data[:, 1] = list(map(reverse_date, data[:, 1]))
        times = (data[:, 1] + "T" + data[:, 2]).astype(np.datetime64)
        data = np.append(np.delete(data,[1,2],1),times.reshape(len(times),1),1)
        titles = np.append(np.delete(titles, [1, 2], 0), np.array(["date-time"]), 0)

        years_data[int(os.path.splitext(data_file)[0])] = data
        years_times[int(os.path.splitext(data_file)[0])] = times
        years_titles[int(os.path.splitext(data_file)[0])] = titles

        data_accuracy = np.zeros(data.shape[1])
        data_accuracy[0] = np.nan
        data_accuracy[-1] = np.nan
        for col in range(1, data.shape[1] - 1):
            c_dat = data[:,col]
            v,a = np.unique(c_dat,return_counts=True)
            most_occurring =v[np.argmax(a)]
            temp = np.abs(c_dat-most_occurring)
            min_dist = np.nanmin(temp[temp != 0])
            data_accuracy[col] = min_dist

        years_data_accuracy[int(os.path.splitext(data_file)[0])] = data_accuracy
loaded_years = list(years_data.keys())
data_confines = list(zip(np.min([np.min(years_data[year],0) for year in years_data.keys()],0),np.max([np.max(years_data[year],0) for year in years_data.keys()],0)))

get_day_of_year = lambda time: (time.astype('timedelta64[m]').astype(float) / (60 * 24))
get_days_of_years = lambda times: get_day_of_year(times - times[0].astype('datetime64[Y]'))
def get_time_from_day(days,accuracy):
    return np.timedelta64(int(days * 60 * 24),'m').astype("timedelta64[{}]".format(accuracy))

def get_accuracy_of_data_index(data_index):
    sample_year = list(years_data.keys())[0]
    return years_data_accuracy[sample_year][data_index]

def get_location_title_of_data():
    sample_year = list(years_data.keys())[0]
    import re
    return re.sub("  +","",years_data[sample_year][0][0])[::-1]
