import numpy as np
import pathlib as pth
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, \
    RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def vec_seg1(array, sub_window_size,
                         overlap=0, clearing_time_index=None, max_time=None):

    if clearing_time_index == None:
        clearing_time_index = sub_window_size

    if max_time == None:
        max_time = array.shape[0]

    stride_size = int(((1 - overlap) * sub_window_size) // 1)
    # print(stride_size)
    start = clearing_time_index - sub_window_size

    sub_windows = (
            start +
            np.expand_dims(np.arange(sub_window_size), 0) +
            # Create a rightmost vector as [0, V, 2V, ...].
            np.expand_dims(np.arange(max_time - sub_window_size + 1, step=stride_size), 0).T
    )

    # Adapted from the work of Syafiq Kamarul Azman, Towards Data Science
    return array[sub_windows]


def pre_process(file, data_in, window_size):
    if file.split("_")[-1] == 'chris.csv':
        data_in = down_sample(data_in, 5)

    rows = data_in.shape[0] - window_size + 1
    cols = data_in.shape[1]
    data_out = np.zeros((rows, cols))
    data_out[:, 0] = data_in[0:rows, 0]

    for ii in range(cols - 1):
        data_out[:, ii + 1] = np.convolve(data_in[:, ii + 1], np.ones(window_size)/window_size, mode='valid')

    if file.partition("_")[0] == 'jumping':
        data_out = np.concatenate((data_out, np.ones((rows, 1))), axis=1)
    else:
        data_out = np.concatenate((data_out, np.zeros((rows, 1))), axis=1)

    return data_out


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
        window = 10
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
                data = pre_process(file, data, window)

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
            data_reformatted[key] = data_reformatted[key][0:size]
            data = data_reformatted[key]
            hdf.create_dataset(key, data=data)

        for name, grp in hdf.items():
            if name != 'dataset':
                fdata_name = name + "_data"
                # print(fdata_name)
                t_data = np.vstack([vec_seg1(np.array(data), 500, 0.4) for data in grp.values()])
                print(fdata_name, ": ", t_data.shape)
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

            fig = plt.figure(name2, figsize=(9.6, 5.4), dpi=200)
            fig.suptitle(name2)
            data = np.array(root_dir[name])[:, 1:5]

            sampling_rate = 100
            N = data.shape[0] // 2 + 1
            n = np.arange(N)
            T = N / sampling_rate
            freq = n / T

            titles = ("Acceleration x FFT", "Acceleration y FFT", "Acceleration z FFT",
                      "Abs Acceleration FFT")

            for ii, ti in enumerate(titles):
                fft_data = np.fft.rfft(data[:, ii])
                ax = fig.add_subplot(len(titles), 1, ii + 1)
                stem_plot = plt.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
                stem_plot[0].set_linewidth(0.4)
                stem_plot[1].set_linewidth(0.4)
                stem_plot[2].set_linewidth(0.4)
                ax.add_container(stem_plot)
                ax.grid(visible=True)
                ax.set_title(ti, fontsize=8)
                ax.set_xlabel('Freq', fontsize=8)
                ax.set_ylabel('Mag', fontsize=8)
                ax.set_xlim((-1, 30))

            fig.set_layout_engine(layout='tight')
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

    print("Returned FFT figures: ", list(figs))
    return figs

def test_train(root_dir, save=None):
    names = list_nodes_t(root_dir, 'd')
    raw_data = np.vstack([root_dir[name] for name in names if (name.split('_')[-1] == 'data')])
    raw_data = raw_data[:, :, 1:6]
    features = pd.DataFrame(columns=['meanx', 'meany', 'meanz', 'medianx', 'mediany', 'medianz', 'varx', 'vary', 'varz', 'action'])

    features['meanx'] = np.mean(np.mean(vec_seg1(raw_data[:, :, 0].T, 50), 1), 0)
    features['meany'] = np.mean(np.mean(vec_seg1(raw_data[:, :, 1].T, 50), 1), 0)
    features['meanz'] = np.mean(np.mean(vec_seg1(raw_data[:, :, 2].T, 50), 1), 0)

    features['medianx'] = np.mean(np.median(vec_seg1(raw_data[:, :, 0].T, 50), 1), 0)
    features['mediany'] = np.mean(np.median(vec_seg1(raw_data[:, :, 1].T, 50), 1), 0)
    features['medianz'] = np.mean(np.median(vec_seg1(raw_data[:, :, 2].T, 50), 1), 0)

    features['varx'] = np.mean(np.var(vec_seg1(raw_data[:, :, 0].T, 50), 1), 0)
    features['vary'] = np.mean(np.var(vec_seg1(raw_data[:, :, 1].T, 50), 1), 0)
    features['varz'] = np.mean(np.var(vec_seg1(raw_data[:, :, 2].T, 50), 1), 0)

    features['action'] = raw_data[:, 1, 4]

    data = features.iloc[:, 1:-1]
    labels = features.iloc[:, -1]

    if save is not None:
        if not os.path.exists(save):
            save.mkdir(parents=True, exist_ok=True)
            print("Made Directory: ", temp)

        save = save / 'features.csv'
        features.to_csv(save, index=False)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)
    dsets = ['dataset/train/x_train', 'dataset/test/x_test', 'dataset/train/y_train', 'dataset/test/y_test']

    for dset, dat in zip(dsets, [x_train, x_test, y_train, y_test]):
        if dset in root_dir:
            del root_dir[dset]
            root_dir.create_dataset(dset, data=dat)
        else:
            root_dir.create_dataset(dset, data=dat)

    return x_train, x_test, y_train, y_test

# TESTING **************************************************************************************************************
# These functions should be called from the loop below tho with commands

# Test Paths but this should be supplied in the while loop through user input or a GUI
# If you want to run this just change that paths to your working directory and recreate
# the directory tree with a test_dir folder and sub-folders for each group and a figures
# folder or similar for saving figure images

p = pth.Path("C:\\Users\\Chris\\PycharmProjects\\Phyphox-390\\test_dir")
p2 = pth.Path("C:\\Users\\Chris\\PycharmProjects\\Phyphox-390\\Figures")
p3 = pth.Path("C:\\Users\\Chris\\PycharmProjects\\Phyphox-390\\Features")

accel_figures = dict()
accel_FFT_figures = dict()
accel_scatter_figures = dict()

# create_hdf5(p)

with h5py.File('./hd5_data.h5', 'r+') as hdf:
    x_train, x_test, y_train, y_test = test_train(hdf, p3)

scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(x_train, y_train)

# Testing the Model
y_pred = clf.predict(x_test)
y_clf_prob = clf.predict_proba(x_test)

# Model Accuracy
acc = accuracy_score(y_test, y_pred)
print('Model Accuracy: ', acc)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
cm_disp = ConfusionMatrixDisplay(cm).plot()
plt.title('Wine Quality Classifier Confusion Matrix')
plt.show()

# F1 Score
f1 = f1_score(y_test, y_pred)
print('Model F1_Score: ', f1)

# ROC and AUC
fp_rate, tp_rate, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_disp = RocCurveDisplay(fpr=fp_rate, tpr=tp_rate).plot()
plt.title('Wine Quality Classifier ROC Curve')
plt.show()

auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print('Model AUC: ', auc)

# with h5py.File('./hd5_data.h5', 'r') as hdf:
#     accel_figures.update(create_accel_plots(hdf, p2))
#     plt.close('all')
#     accel_scatter_figures.update(accel_scatter_plots(hdf, p2))
#     plt.close('all')
#     accel_FFT_figures.update(accel_fft_plots(hdf, p2))



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
