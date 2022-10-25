from cycler import cycler
import matplotlib.pyplot as plt

from load_data import get_days_of_years,years_times,fix_hebrew_for_print,years_data,years_titles,get_time_from_day,get_day_of_year,years_data_accuracy,data_confines

def plot_wether_graph(year,*q_indexes,b_x_is_day=False,kwargs_arr=None):
    colors_cycler = cycler("color",[(0.8,0.1,0.1,0.7),(0.1,0.8,0.1,0.7),(0.1,0.1,0.8,0.7),(0.1,0.7,0.7,0.7),(0.7,0.1,0.7,0.7),(0.3,0.3,0.3,0.7)])()
    _,ax1 = plt.subplots(1, 1)
    ax1.tick_params(axis='x')
    axes = [ax1]
    for q in q_indexes:
        if q != q_indexes[0]:
            axes.append(ax1.twinx())
            axes[-1].spines["right"].set_position(("axes", 1 + (len(axes) - 2) *0.06))
        plt_cycled_graphics = next(colors_cycler)
        if b_x_is_day:
            T = get_days_of_years(years_times[year])
        else:
            T = years_times[year]
        passed_kw = kwargs_arr[len(axes) - 1] if kwargs_arr is not None and kwargs_arr[len(axes) - 1] is not None else {}
        axes[-1].plot(T, years_data[year][:, q],**plt_cycled_graphics, **passed_kw)
        color = plt_cycled_graphics["color"]
        axes[-1].set_ylabel(fix_hebrew_for_print(years_titles[year][q]))
        axes[-1].yaxis.label.set_color(color)
        axes[-1].tick_params(axis='y', colors=color)
    if b_x_is_day:
        axes[0].set_xlabel("Day of the year")
    else:
        axes[0].set_xlabel("Date of the year")
    plt.show()

if __name__ == "__main__":
    plot_wether_graph(2017,1,2,3)