import numpy as np
import pandas as pd
import antropy as ant
from scipy.signal import butter, lfilter
from yasa import bandpower
from tsflex.processing import SeriesPipeline, SeriesProcessor


def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, sig)
    return y


def filter_data(data_records, freq_eeg, freq_acc, axes_names):
    # Define filtering operations to perform on EEG/acc data
    eeg_bandpass = SeriesProcessor(
        function=butter_bandpass_filter,
        series_names=[f"EEG{i}" for i in range(1, 6)],
        lowcut=0.4,
        highcut=30,
        fs=freq_eeg,
    )

    acc_bandpass_breathing = SeriesProcessor(
        function=butter_bandpass_filter,
        series_names=[f"Accelerometer_{axes_names[i]}" for i in range(3)],
        lowcut=0.08,
        highcut=0.3,
        fs=freq_acc,
    )

    acc_bandpass_heartbeat = SeriesProcessor(
        function=butter_bandpass_filter,
        series_names=[f"Accelerometer_{axes_names[i]}" for i in range(3)],
        lowcut=0.6,
        highcut=1.7,
        fs=freq_acc,
    )

    process_pipe = SeriesPipeline(
        [
            eeg_bandpass,
            acc_bandpass_breathing,
        ]
    )
    # Second pipeline because we are filtering again accelerator data
    process_pipe_bis = SeriesPipeline(
        [
            acc_bandpass_heartbeat,
        ]
    )

    # Perform the filtering
    for record in data_records:
        A = process_pipe.process(data_records[record], return_all_series=True)
        B = process_pipe_bis.process(data_records[record][-3:], return_all_series=True)
        for i, s in enumerate(B):
            s.rename(f"Accelerometer_h_{axes_names[i]}", inplace=True)
        data_records[record] = A + B
        del A, B

    return data_records


def wrapped_higuchi_fd(x):
    x = np.array(x, dtype="float64")
    return ant.higuchi_fd(x)

def wrapped_bandpowers(x, sf, bands):
    return bandpower(x, sf=sf, bands=bands).values[0][:-2]

def compute_features(data_records, feature_collection):
    for records in data_records:
        data_records[records] = feature_collection.calculate(
            data_records[records], return_df=True, show_progress=True
        ).astype("float32")
    return data_records


def add_eeg_bands_features(df_feats, bands):
    for eeg_sig in [f"EEG{i}" for i in range(1, 6)]:
        eeg_bands = [
            c
            for c in df_feats.columns
            if c.startswith(eeg_sig) and c.split("__")[1] in bands
        ]
        windows = sorted(set(b.split("__")[-1] for b in eeg_bands))
        for window in windows:
            # Select the spectral powers
            delta = (
                df_feats["__".join([eeg_sig, "sdelta", window])]
                + df_feats["__".join([eeg_sig, "fdelta", window])]
            )
            fdelta_theta = (
                df_feats["__".join([eeg_sig, "fdelta", window])]
                + df_feats["__".join([eeg_sig, "theta", window])]
            )
            alpha = df_feats["__".join([eeg_sig, "alpha", window])]
            beta = df_feats["__".join([eeg_sig, "beta", window])]
            theta = df_feats["__".join([eeg_sig, "theta", window])]
            sigma = df_feats["__".join([eeg_sig, "sigma", window])]
            # Calculate the ratios
            df_feats[
                "__".join([eeg_sig, "fdelta+theta", window])
            ] = fdelta_theta.astype("float32")

            df_feats["__".join([eeg_sig, "alpha/theta", window])] = (
                alpha / theta
            ).astype("float32")
            df_feats["__".join([eeg_sig, "delta/beta", window])] = (
                delta / beta
            ).astype("float32")
            df_feats["__".join([eeg_sig, "delta/sigma", window])] = (
                delta / sigma
            ).astype("float32")
            df_feats["__".join([eeg_sig, "delta/theta", window])] = (
                delta / theta
            ).astype("float32")
    return df_feats


def add_shifted_feats(df_feats):
    feats_30s = [f for f in df_feats.columns if "w=30s" in f]
    feats_60s = [f for f in df_feats.columns if "w=1m_" in f]
    feats_90s = [f for f in df_feats.columns if "w=1m30s" in f]
    dfs = []
    for record in df_feats.record.unique():
        sub_df = df_feats[df_feats.record == record]

        sub_df = sub_df.merge(
            sub_df[feats_90s].shift(1).add_suffix("_shift=30s"),
            left_index=True,
            right_index=True,
        )
        sub_df = sub_df.drop(columns=feats_90s)

        sub_df = sub_df.merge(
            sub_df[feats_60s].shift(1).add_suffix("_shift=30s"),
            left_index=True,
            right_index=True,
        )

        sub_df = sub_df.merge(
            sub_df[feats_30s].shift(2).add_suffix("_shift=1m"),
            left_index=True,
            right_index=True,
        )
        sub_df = sub_df.merge(
            sub_df[feats_30s].shift(1).add_suffix("_shift=30s"),
            left_index=True,
            right_index=True,
        )
        sub_df = sub_df.merge(
            sub_df[feats_30s].shift(-1).add_suffix("_shift=-30s"),
            left_index=True,
            right_index=True,
        )
        sub_df = sub_df.merge(
            sub_df[feats_30s].shift(-2).add_suffix("_shift=-1m"),
            left_index=True,
            right_index=True,
        )
        dfs += [sub_df]
    df_feats = pd.concat(dfs)
    return df_feats
