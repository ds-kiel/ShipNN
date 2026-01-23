
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.signal import resample
import soundfile as sf
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def _combine_data_splits(split_1, split_2, split_3):
    df_1 = pd.read_csv(split_1,index_col=0)
    df_2 = pd.read_csv(split_2,index_col=0)
    df_3 = pd.read_csv(split_3,index_col=0)

    df_1['scenario']="inclusion_2000_exclusion_4000"
    df_2['scenario']="inclusion_3000_exclusion_5000"
    df_3['scenario']="inclusion_4000_exclusion_6000"

    df_concat = pd.concat([df_1,df_2,df_3], axis=0)
    return df_concat

def _remove_common_data(data):
    label_to_min_subset = data.groupby('MMSI')['scenario'].min()
    data = data[data.apply(lambda row: row['scenario'] == label_to_min_subset[row['MMSI']], axis=1)]
    return data

def _group_data_by_MMSI(data):
    data_copy = data.copy()
    data_copy['no_of_instances'] = 1
    data_copy = data_copy[['no_of_instances','MMSI']]
    unique_ships = len(data_copy['MMSI'].value_counts())
    print("Total unique ships/ MMSI values",unique_ships )
    group_by_MMSI = data_copy.groupby(['MMSI']).sum()
    group_by_MMSI = group_by_MMSI.sort_values(by='no_of_instances', ascending=False).reset_index()
    return group_by_MMSI

def _make_subset(root_dir,no_of_ships):
    root_dir_2to4k = f"{root_dir}/inclusion_2000_exclusion_4000/ShipNN_metadata.csv"
    root_dir_3to5k = f"{root_dir}/inclusion_3000_exclusion_5000/ShipNN_metadata.csv"
    root_dir_4to6k = f"{root_dir}/inclusion_4000_exclusion_6000/ShipNN_metadata.csv"

    data = _combine_data_splits(root_dir_2to4k,root_dir_3to5k,root_dir_4to6k)
    data = _remove_common_data(data)
    MMSI_vs_instances = _group_data_by_MMSI(data)
    
    expanded_rows = []
    for _, row in MMSI_vs_instances.iterrows():
        for trace in range(1, int(row['no_of_instances']) + 1):
            expanded_rows.append({'MMSI': row['MMSI'], 'no_of_instances': trace})

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df.groupby('no_of_instances')['MMSI'].count().reset_index()
    expanded_df.columns = ['no_of_instances', 'no_of_ships']
    df_instances_vs_ship = expanded_df.groupby(by='no_of_ships', as_index=False).max()
    threshold = df_instances_vs_ship[df_instances_vs_ship['no_of_ships']==no_of_ships].head(1)['no_of_instances'].values[0]
    
    ships_with_threshold_instances = MMSI_vs_instances[MMSI_vs_instances["no_of_instances"]>=threshold]
    all_data_for_no_of_ships=data[data["MMSI"].isin(list(ships_with_threshold_instances["MMSI"]))]
    final_data=all_data_for_no_of_ships.sample(frac=1).groupby('MMSI').head(threshold)
    
    le = preprocessing.LabelEncoder()
    le.fit(final_data.MMSI)
    final_data['categorical_label'] = le.transform(final_data.MMSI)
    train_files, test_files = train_test_split(final_data, test_size=0.1, random_state=42,stratify=[item for item in final_data.MMSI])
    return train_files, test_files



# -------- Waveform Augmentations --------
def add_noise(waveform, noise_factor=0.005):
    noise = tf.random.normal(shape=tf.shape(waveform), mean=0.0, stddev=1.0)
    return waveform + noise_factor * noise

def time_shift(waveform, shift_max=0.2):
    """Randomly shift in time, up to 20% of length"""
    shift = tf.random.uniform([], -shift_max, shift_max) * tf.cast(tf.shape(waveform)[0], tf.float32)
    shift = tf.cast(shift, tf.int32)
    return tf.roll(waveform, shift, axis=0)

# -------- Spectrogram Augmentations --------
def spec_augment(spectrogram, num_masks=2, freq_mask_param=10, time_mask_param=15):
    """Applies SpecAugment: random frequency & time masking"""
    for _ in range(num_masks):
        # Frequency mask
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.shape(spectrogram)[0] - f, dtype=tf.int32)
        spectrogram = tf.concat([
            spectrogram[:f0, :, :],
            tf.zeros_like(spectrogram[f0:f0+f, :, :]),
            spectrogram[f0+f:, :, :]
        ], axis=0)

        # Time mask
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.shape(spectrogram)[1] - t, dtype=tf.int32)
        spectrogram = tf.concat([
            spectrogram[:, :t0, :],
            tf.zeros_like(spectrogram[:, t0:t0+t, :]),
            spectrogram[:, t0+t:, :]
        ], axis=1)

    return spectrogram

def unpad_mfcc(padded_mfcc):
    
    return padded_mfcc[..., :-82, :]

def pad_mfcc_with_mean(mfcc_out, pad_size=82):
    
    mean_vals = mfcc_out.mean(axis=-2, keepdims=True)
    pad_block = np.repeat(mean_vals, pad_size, axis=-2)
    return np.concatenate([mfcc_out, pad_block], axis=-2)


def _parse_function(root_dir, file_path, split_type, file_index, ship_type, MMSI, scenario, categorical_label, transform, target_sample_rate):
    """Parses a single data point (file path and label) to a tensor."""
    
    # Convert tensor values to native Python types
    root_dir = str(root_dir.numpy().decode('utf-8'))
    file_path = str(file_path.numpy().decode('utf-8'))
    split_type = str(split_type.numpy().decode('utf-8'))
    file_index = int(file_index.numpy())
    ship_type = str(ship_type.numpy().decode('utf-8'))
    MMSI = str(MMSI.numpy())
    scenario = str(scenario.numpy().decode('utf-8'))
    categorical_label = categorical_label.numpy()

    if len(transform)>0:
        transform = [str(t.numpy().decode('utf-8')) for t in transform]

        # Load and stack pre-processed .npy files
        waveform_list = []
        for preprocessing_type in transform:
            path = os.path.join(root_dir, scenario, split_type, preprocessing_type, ship_type, str(file_index) +'.npy')
            data = np.load(path).squeeze(0)
            if preprocessing_type == 'mfcc':
                data=unpad_mfcc(data)
                data=pad_mfcc_with_mean(data)
            m = np.mean(data)
            s = np.std(data)
            normed = (data - m) / (s + 1e-7)
            waveform_list.append(normed)
        
        waveform = np.stack(waveform_list)  # Shape: (freq_bins, time_steps, num_transforms)
        waveform=np.transpose(waveform, (1,2,0))  # Shape: (time_steps, freq_bins, num_transforms)
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    else:
        # # Load and resample raw .wav files
        audio_path = os.path.join(root_dir, scenario, split_type, "audio", ship_type, str(file_index)+'.wav')

        waveform, sample_rate = sf.read(audio_path)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        target_sample_rate = tf.cast(target_sample_rate, dtype=tf.int64)
        
        # Resample if necessary
        if sample_rate != target_sample_rate:
            waveform = resample(waveform, target_sample_rate)
            
        waveform = tf.cast(waveform, tf.float32)
        waveform = tf.reshape(waveform, [1, -1, 1])
    categorical_label = tf.convert_to_tensor(categorical_label, dtype=tf.int64)

    return waveform, categorical_label, file_path

def create_tf_dataset(root_dir, files_df, transform, target_sample_rate, batch_size, shuffle=False, apply_augmentation=False,augmentation_type="noise"):
    """Creates a tf.data.Dataset from a pandas DataFrame."""
    
    dataset = tf.data.Dataset.from_tensor_slices(dict(files_df))

    def _parser_wrapper(file_info):
        waveform, label,file_path = tf.py_function(
            _parse_function,
            [
                root_dir,
                file_info['file_path'],
                file_info['split_type'],
                file_info['file_index'],
                file_info['ship_type'],
                file_info['MMSI'],
                file_info['scenario'],
                file_info['categorical_label'],
                transform,
                target_sample_rate
            ],
            [tf.float32, tf.int64, tf.string]
        )
        if transform and len(transform) > 0:
            if apply_augmentation:
                waveform = spec_augment(waveform)
            waveform.set_shape([95, 126, len(transform)])
        else:
            waveform = tf.reshape(waveform, (target_sample_rate,))
            if apply_augmentation:
                if augmentation_type=="noise":
                    waveform = add_noise(waveform)
                elif augmentation_type=="timeshift":
                    waveform = time_shift(waveform)

        label.set_shape([])
        
        # print("Final waveform shape:", waveform.shape)
        # print("Final label shape:", label.shape)

        return waveform, label

    # Map the parsing function over the dataset
    dataset = dataset.map(_parser_wrapper)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(files_df))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

def get_datasets(config):
    """
    Main function to get data loaders.
    """
    train_files, test_files = _make_subset(
        config["dataset"]["root_dir"], config["model"]["num_classes"]
    )
    
    print("Creating test datasets")
    # Create the various testing datasets
    test_dataset_head = create_tf_dataset(
        root_dir=config["dataset"]["root_dir"],
        files_df=test_files,
        transform=config["dataset"]["preprocess"],
        target_sample_rate=config["dataset"]["sample_rate"],
        batch_size=config["dataset"]["batch_size"]
    )

    records_per_category = test_files.scenario.value_counts().min()
    test_files_same = test_files.groupby("scenario").apply(lambda x: x.sample(n=min(len(x), records_per_category))).reset_index(drop=True)
    
    test_dataset_same = create_tf_dataset(
        root_dir=config["dataset"]["root_dir"],
        files_df=test_files_same,
        transform=config["dataset"]["preprocess"],
        target_sample_rate=config["dataset"]["sample_rate"],
        batch_size=config["dataset"]["batch_size"]
    )
    
    test_dataset_s1 = create_tf_dataset(
        root_dir=config["dataset"]["root_dir"],
        files_df=test_files[test_files.scenario=="inclusion_2000_exclusion_4000"],
        transform=config["dataset"]["preprocess"],
        target_sample_rate=config["dataset"]["sample_rate"],
        batch_size=config["dataset"]["batch_size"]
    )
    
    test_dataset_s2 = create_tf_dataset(
        root_dir=config["dataset"]["root_dir"],
        files_df=test_files[test_files.scenario=="inclusion_3000_exclusion_5000"],
        transform=config["dataset"]["preprocess"],
        target_sample_rate=config["dataset"]["sample_rate"],
        batch_size=config["dataset"]["batch_size"]
    )
    
    test_dataset_s3 = create_tf_dataset(
        root_dir=config["dataset"]["root_dir"],
        files_df=test_files[test_files.scenario=="inclusion_4000_exclusion_6000"],
        transform=config["dataset"]["preprocess"],
        target_sample_rate=config["dataset"]["sample_rate"],
        batch_size=config["dataset"]["batch_size"]
    )

    return train_files, test_files, [test_dataset_head, test_dataset_s1, test_dataset_s2, test_dataset_s3]