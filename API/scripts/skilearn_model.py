import librosa
import pandas as pd
import itertools
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def find_anomalies_audio(path_to_audio):
    y_test, sr = librosa.load(path_to_audio)

    y_test_df = pd.DataFrame.from_dict({'Audio': y_test})

    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)

    outliers_fraction = float(.0007)
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    y_test_df['anomaly'] = model.predict(data)

    return y_test_df, sr


def postproccess_annomalies(y_test_df, sr):
    window = sr // 5

    y_test_df['anomaly'] = (y_test_df['anomaly'] - 1) * -1 / 2

    y_test_df['smoothed'] = y_test_df['anomaly'].rolling(window=window, min_periods=1).max().astype('int')
    y_test_df['smoothed_rev'] = y_test_df['anomaly'].iloc[::-1].rolling(window=window, min_periods=1).max().astype(
        'int').iloc[::-1]
    y_test_df['smoothed'] = y_test_df['smoothed'] * y_test_df['smoothed_rev']

    smoothed_list = y_test_df['smoothed'].tolist()
    intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(smoothed_list)]

    # get index of intervals >= 0.3s
    min_frames = sr * (len(y_test_df)) * 7.54e-08

    starting_times = []
    starting_seconds = []
    for idx, interval in enumerate(intervals):
        if interval[0] == 1 and interval[1] >= min_frames:
            starting_frame = 0
            if idx == 0:
                starting_times.append('00:00')
            else:
                for i in range(idx):
                    starting_frame += intervals[i][1]
                num_seconds = int(starting_frame / sr)
                if not starting_seconds:
                    starting_seconds.append([num_seconds, num_seconds])
                elif (num_seconds - starting_seconds[-1][1]) <= 4:
                    starting_seconds[-1] = [starting_seconds[-1][0], num_seconds]
                else:
                    starting_seconds.append([num_seconds, num_seconds])

                starting_time = str(datetime.timedelta(seconds=num_seconds))[-5:]
                starting_times.append(starting_time)

    # если ничего не нашли
    if not starting_seconds:
        anomaly_list = y_test_df['anomaly'].astype('int').tolist()
        intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(anomaly_list)]

        all_lenght = [x[1] for x in intervals]
        longest_three = sorted(all_lenght, reverse=True)[:3]

        starting_times = []
        starting_seconds = []
        for idx, interval in enumerate(intervals):
            if interval[0] == 1 and interval[1] in longest_three:
                starting_frame = 0
                if idx == 0:
                    starting_times.append('00:00')
                else:
                    for i in range(idx):
                        starting_frame += intervals[i][1]
                    num_seconds = int(starting_frame / sr)
                    if not starting_seconds:
                        starting_seconds.append([num_seconds, num_seconds])
                    elif (num_seconds - starting_seconds[-1][1]) <= 4:
                        starting_seconds[-1] = [starting_seconds[-1][0], num_seconds]
                    else:
                        starting_seconds.append([num_seconds, num_seconds])

                    starting_time = str(datetime.timedelta(seconds=num_seconds))[-5:]
                    starting_times.append(starting_time)

    intervals = []
    for pair in starting_seconds:
        intervals.append(
            [(datetime.timedelta(seconds=(pair[0] - 3))), (datetime.timedelta(seconds=(pair[1] + 6))), 'ML'])

    starting_times = list(set(starting_times))
    starting_times.sort()
    return intervals, starting_times
