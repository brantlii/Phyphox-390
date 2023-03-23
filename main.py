import numpy as np
import pathlib as pth
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


def list_nodes_t(root_dir, n_type=None):
    # Visits each node in an HDF file and checks if the node is a dataset or group.
    # If the node is a dataset or group, the name of the node is appended to a list
    # The list is then returned

    list_of_names = []
    root_dir.visit(list_of_names.append)
    ret = []

    if n_type == ('d' or 'D'):
        for name in list_of_names:
            if isinstance(root_dir[name], h5py.Dataset):
                ret.append(name)
            else:
                pass

    elif n_type == ('g' or 'G'):
        for name in list_of_names:
            if isinstance(root_dir[name], h5py.Group):
                ret.append(name)
            else:
                pass

    elif n_type is None:
        ret = list_of_names

    else:
        print("Argument 'n_type' in list_nodes_t(root_dir, n_type) must be either 'd', 'g', or left blank.")

    return ret


def dataset_visitor(root_dir, func, args=None):
    # Allows application of a function to every dataset in the HDF file.
    #
    # root_dir is the HDF5 file (in other words root group) to operate upon
    #
    # func is a callable function that will be applied to every dataset found
    #
    # args=None is an optional tuple of arguments that the callable function requires
    #
    # The callable function must have the signature function(root_dir, name, *args) where name is the name of the
    # current HDF5 dataset being operated upon, root_dir is the HDF5 file (in other words root group) of which the
    # dataset is a member, and *args specifies any other arguments needed by the callable function

    nodes = list_nodes_t(root_dir, 'd')

    for name in nodes:
        if args is None:
            func(root_dir, name)
        else:
            func(root_dir, name, *args)


def truncate(root_dir, dname, t_size, t_dir):
    if isinstance(t_size, int):
        shape = root_dir[dname].shape

        if t_dir == 'rows':
            if shape[0] < t_size:
                print("Fewer rows than truncation size!")
            else:
                d1 = root_dir[dname]
                d1 = d1[0:t_size, :]

        elif t_dir == 'cols':
            if shape[1] < t_size:
                print("Fewer columns than truncation size!")
            else:
                d1 = root_dir[dname]
                d1 = d1[:, 0:t_size]
        else:
            print("Please specify a truncation direction with the argument 'rows' or 'cols'")

        del root_dir[dname]
        root_dir.create_dataset(dname, data=d1)
    else:
        print("Please specify an integer truncation size")


def down_sample(arr, rate):
    # Given a numpy array keep 1 out of every X rows

    d1 = []

    if not isinstance(rate, int):
        print("Please specify an integer downsampling rate with the rate arg")
    else:
        d1 = arr[0::rate]
    return d1


def reformat(file, data):
    if file.partition("_")[0] == 'jumping':
        action = np.ones((data.shape[0], 1))
    else:
        action = np.zeros((data.shape[0], 1))

    data = np.concatenate((data, action), axis=1)

    if file.split("_")[-1] == 'chris.csv':
        data = down_sample(data, 5)

    return data


def overwrite_needed(dataset_name, datasets):
    ret = 'n'
    if dataset_name in datasets:
        ret = 's'
        res = input("\"" + dataset_name + "\" already exists! Overwrite the dataset? [y/n]")
        if res == ('y' or 'Y'):
            ret = 'o'
        if res != ('n' or 'N'):
            print("Bad input, skipping the dataset.")
    return ret


def create_hdf5(path):
    # Given a path to a root directory, creates groups for every folder and subfolder and creates datasets for every csv
    # Saves the hdf5 file to the current working directory

    groups = set()
    datasets = set()

    with h5py.File('./hd5_data.h5', 'w') as hdf:
        # test overwrite
        # testdata = np.random.random(size=(1000,1000))
        # groups.add('/chris')
        # datasets.add('/chris/walking_hand_chris.csv')
        # hdf.create_dataset('/chris/walking_hand_chris.csv', data=testdata)
        size = 0
        data_reformatted = dict()

        for root, dirs, files in os.walk(path):

            for direct in dirs:
                gr_name = root[len(str(path)):].replace("\\", "/") + "/" + direct

                if gr_name not in groups:
                    hdf.create_group(gr_name)
                    groups.add(gr_name)
                    # print(group_name)

            for file in files:
                dataset_name = root[len(str(path)):].replace("\\", "/") + "/" + file

                data = np.genfromtxt(os.path.join(root, file), skip_header=1, delimiter=',')
                data = reformat(file, data)

                if size == 0:
                    size = data.shape[0]

                if size > data.shape[0]:
                    size = data.shape[0]

                # print(dataset_name)

                t = overwrite_needed(dataset_name, datasets)
                match t:
                    case 'n':
                        datasets.add(dataset_name)
                        data_reformatted.update({dataset_name: data})
                    case 'o':
                        del hdf[dataset_name]
                        data_reformatted.update({dataset_name: data})
                    case 's':
                        pass

                size = (size // 1000) * 1000

        # print(list(data_reformatted.keys()))
        for key, value in data_reformatted.items():
            temp = value[0:size]
            data_reformatted[key] = temp
            # print(temp.shape)
            hdf.create_dataset(key, data=temp)

        for name, grp in hdf.items():
            if name != 'dataset':
                fdata_name = name + "_data"
                # print(fdata_name)
                t_data = np.hstack(
                    [arr.reshape((size // 10), 1, 6) for data in grp.values() for arr in np.array_split(data, 10)])
                grp.create_dataset(fdata_name, data=t_data)

                # for old_data in grp.keys():
                #     if old_data != fdata_name:
                #         del grp[old_data]

        print("Created HDF file with the following groups: ", list(hdf.items()), "\n")


def create_accel_plots(root_dir, save=None):
    figs = dict()
    names = list_nodes_t(root_dir, 'd')

    for name in names:
        if name.split("_")[1] != 'data':
            print("Dataset! " + "/" + name)

            data = np.array(root_dir[name])
            fig = plt.figure(name, figsize=(9.6, 5.4))
            fig.suptitle(name)
            fig.canvas.manager.set_window_title(name)

            titles = ("Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)")

            # Simply returns the data in the HDF5 datasets for now
            # Add wrapper or delegate to do stuff to the data and call this again
            # Or possibly make a copy of the data and modify using dataset_visitor

            for i, ti in zip(range(len(titles)), titles):
                ax = fig.add_subplot(3, 1, i + 1)
                ax.plot(data[0:, 0], data[0:, i + 1], linewidth=0.4)
                ax.grid(visible=True)
                ax.set_title(ti, fontsize=8)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('m/s^2', fontsize=8)

            fig.tight_layout()

            figs.update({name: fig})

            if save is not None:
                # print(root_dir[name].parent.name)
                temp = save / root_dir[name].parent.name[1:]
                # print(temp)

                if not os.path.exists(temp):
                    temp.mkdir(parents=True, exist_ok=True)
                    print("Made Directory: ", temp)

                temp = save / name.replace("csv", "png")
                plt.savefig(str(temp).replace("\\", "/"), dpi=200)

    print("Returned figures: ", list(figs))
    return figs


def accel_scatter_plots(root_dir, save=None):
    figs = dict()
    names = list_nodes_t(root_dir, 'd')

    for name in names:
        if name.split("_")[1] != 'data':
            print("Dataset! " + "/" + name)
            data = np.array(root_dir[name])[:, 1:5]
            df = pd.DataFrame(data, columns=['Acceleration x (m/s^2)',
                                             'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                                             'Abs Acceleration (m/s^2)'])

            fig, axs = plt.subplots(4, 4, figsize=(38.4, 21.6))
            fig.set_label(name)
            fig.suptitle(name)
            fig.canvas.manager.set_window_title(name)

            axes = pd.plotting.scatter_matrix(df, ax=axs[0:16], grid=True, diagonal='hist',
                                              marker='.', s=10)

            fig.tight_layout()
            figs.update({name.replace(".csv", "_scatter"): fig})

            if save is not None:
                # print(root_dir[name].parent.name)
                temp = save / root_dir[name].parent.name[1:]
                # print(temp)

                if not os.path.exists(temp):
                    temp.mkdir(parents=True, exist_ok=True)
                    print("Made Directory: ", temp)

                temp = save / name.replace(".csv", "_scatter.png")
                plt.savefig(str(temp).replace("\\", "/"), dpi=100)

    print("Returned figures: ", list(figs))
    return figs


def accel_fft_plots(root_dir, save=None):
    figs = dict()
    names = list_nodes_t(root_dir, 'd')

    for name in names:
        if name.split("_")[1] != 'data':
            name2 = name.replace(".csv", "_FFT")
            print("Dataset! " + "/" + name2)

            data = np.array(root_dir[name])[:, 1:5]

            size = data.shape[0]//2 + 1
            new_data = np.zeros((size, data.shape[1]))

            for i in range(data.shape[1]):
                new_data[:, i] = np.fft.rfft(data[:, i])

            fig = plt.figure(name2, figsize=(9.6, 5.4))
            fig.suptitle(name2)
            fig.canvas.manager.set_window_title(name2)

            titles = ("Acceleration x FFT", "Acceleration y FFT", "Acceleration z FFT",
                      "Abs Acceleration FFT")

            s_rate = 100
            t = 5000/s_rate
            n = np.arange(size)
            freq = n/t

            for ii, ti in enumerate(titles):
                ax = fig.add_subplot(len(titles), 1, ii + 1)
                ax.stem(freq, np.abs(new_data[0:, ii]), 'b', markerfmt=" ", basefmt="-b")
                ax.grid(visible=True)
                ax.set_title(ti, fontsize=8)
                ax.set_xlabel('Freq', fontsize=8)
                ax.set_ylabel('Mag', fontsize=8)
                ax.set_xlim((-1, 8))

            fig.tight_layout()
            figs.update({name2: fig})

            if save is not None:
                # print(root_dir[name].parent.name)
                temp = save / root_dir[name].parent.name[1:]
                # print(temp)

                if not os.path.exists(temp):
                    temp.mkdir(parents=True, exist_ok=True)
                    print("Made Directory: ", temp)

                temp = save / name.replace(".csv", "_FFT.png")
                plt.savefig(str(temp).replace("\\", "/"), dpi=200)

    print("Returned figures: ", list(figs))
    return figs


# TESTING **************************************************************************************************************
# These functions should be called from the loop below tho with commands

# Test Paths but this should be supplied in the while loop through user input or a GUI
# If you want to run this just change that paths to your working directory and recreate
# the directory tree with a test_dir folder and sub-folders for each group and a figures
# folder or similar for saving figure images

p = pth.Path("C:\\Users\\Chris\\PycharmProjects\\ELEC390_Project\\test_dir")
p2 = pth.Path("C:\\Users\\Chris\\PycharmProjects\\ELEC390_Project\\Figures")

# accel_figures = dict()
# accel_FFT_figures = dict()
# accel_scatter_figures = dict()
#
# create_hdf5(p)
with h5py.File('./hd5_data.h5', 'r') as hdf:
    print(list_nodes_t(hdf, 'd'))

    d1 = hdf.get('brant/walking_back_pocket_brant.csv')
    d1 = np.array(d1)

    # print(d1)
    print(np.transpose(d1[:, 1]))
    d2 = np.convolve(d1[:, 1], np.ones(10)/10, mode='same')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(4,1, 1)
    ax1.plot(d1[:, 0], d1[:, 1], linewidth=0.4)

    d1_fft = np.fft.rfft(d1[:, 1])

    size = d1[:, 1].shape[0] // 2 + 1
    s_rate = 100
    t = 5000 / s_rate
    n = np.arange(size)
    freq = n / t

    ax2 = fig1.add_subplot(4, 1, 2)
    ax2.stem(freq, np.abs(d1_fft), 'b', markerfmt=" ", basefmt="-b")
    ax2.set_xlim((6, 12))
    ax2.set_ylim((0, 2000))

    ax3 = fig1.add_subplot(4, 1, 3)
    ax3.plot(d1[:, 0], d2, linewidth=0.4)

    d2_fft = np.fft.rfft(d2)

    size = d2.shape[0] // 2 + 1
    s_rate = 100
    t = d2.shape[0] / s_rate
    n = np.arange(size)
    freq = n / t

    ax4 = fig1.add_subplot(4, 1, 4)
    ax4.stem(freq, np.abs(d2_fft), 'b', markerfmt=" ", basefmt="-b")
    ax4.set_xlim((6, 12))
    ax4.set_ylim((0, 2000))
    plt.show()

    # fig, ax = plt.subplots(1, 1, linewidth=0.4)
    # ax.plot(np.transpose)
# with h5py.File('./hd5_data.h5', 'r') as hdf:
#
    # accel_figures.update(create_accel_plots(hdf, p2))
    # plt.close('all')
    # accel_scatter_figures.update(accel_scatter_plots(hdf, p2))
    # plt.close('all')
    # accel_FFT_figures.update(accel_fft_plots(hdf, p2))



# TESTING **************************************************************************************************************

# Using "Match, Case" control flow here to run the functions from the console with text commands
# emulates the kind of standalone interactive environment we'll be in with a desktop app
#
# while 1 > 0:
#     match input()
#       case
#           func()
#       ...
#       case
#           do smtg else
#     ls = list(figures.keys())
#     print("\n", ls)


# Just as a reminder for naming
#     sets = ['walking_front_pocket', 'walking_hand', 'walking_back_pocket',
#             'walking_jacket_no_snow', 'walking_backpack', 'walking_jacket_snow',
#             'stairs_front_pocket', 'jogging_hand', 'jumping_stationary_hand', 'jumping_back_pocket']
