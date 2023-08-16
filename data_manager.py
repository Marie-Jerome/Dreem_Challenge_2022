import os
import numpy as np
import pandas as pd
from collections import defaultdict


def load_data(input_folder, freq_eeg, freq_acc, axes_names):
    data_records = defaultdict(list)
    for record in os.listdir(input_folder):
        if not record.endswith(".npy"):
            continue
        df = np.load(os.path.join(input_folder, f"{record}"))

        # for all EEGs; one channel = 7500 columns
        for i in range(5):
            eeg = df[
                :, 1 + 7500 * i : 7500 * (i + 1) + 1
            ].flatten()  # flatten the measurements
            idx = pd.date_range(
                start=0, periods=len(eeg), freq=pd.Timedelta(1 / freq_eeg, unit="s")
            )  # date each measurements
            eeg = pd.Series(
                eeg, index=idx, name=f"EEG{i+1}"
            )  # save into a Series data frame
            data_records[record] += [eeg]  # add to the data list

        # for all accelerometers
        for i in range(3):
            acc = df[
                :, 37501 + i * 1500 : 37501 + (i + 1) * 1500
            ].flatten()  # flatten the measurements
            idx = pd.date_range(
                start=0, periods=len(acc), freq=pd.Timedelta(1 / freq_acc, unit="s")
            )  # date each measurements
            acc = pd.Series(
                acc, index=idx, name=f"Accelerometer_{axes_names[i]}"
            )  # save into a Series data frame
            data_records[record] += [acc]  # add to the data list

    return data_records


def add_labels(data_records, train_folder):
    hypnograms = pd.read_csv(os.path.join(train_folder, "targets_train.csv"))

    for record in data_records:
        record_number = int(record[-5])
        idx = data_records[record].index
        label = (
            pd.DataFrame(hypnograms[hypnograms["record"] == record_number]["target"])
            .iloc[:-1]
            .set_index(idx)
        )
        data_records[record] = data_records[record].merge(
            label, left_index=True, right_index=True
        )

    return data_records


def add_record_and_patient_ids(data_records):
    df_feats = []

    for record in data_records.keys():
        data_records[record]["record"] = record
        data_records[record]["patient_id"] = record[-5]
        df_feats += [data_records[record]]
    df_feats = pd.concat(df_feats)
    
    return df_feats
