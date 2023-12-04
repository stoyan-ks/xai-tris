from typing import List, Dict

import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.ndimage import gaussian_filter
from common import DataRecord, DataScenarios, SEED
from utils import load_json_file, dump_as_pickle
from data.data_utils import sphering, generate_backgrounds, generate_fixed, generate_translations_rotations, generate_xor, normalise_data, optimal_signal_preserving_whitening, partial_regression, scale_to_bound, symmetric_orthogonalization, cholesky_whitening

from datetime import datetime
from pathlib import Path

import os
import sys

# Required imports for the modified function
from typing import List, Dict
import os

os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Updated whitening methods dictionary
whitening_methods = {
    "symmetric_orthogonalization": symmetric_orthogonalization,
    "sphering": sphering,
    "partial_regression": partial_regression,
    "cholesky_whitening": cholesky_whitening,
    "optimal_signal_preserving_whitening": optimal_signal_preserving_whitening
}

scenario_dict = {
    'linear': generate_fixed,
    'multiplicative': generate_fixed,
    'translations_rotations': generate_translations_rotations,
    'xor': generate_xor
}

captured_data = {
}

def whitened_data_generation(config: Dict, output_dir: str) -> List:
    image_shape = np.array(config["image_shape"]) * config["image_scale"]
    
    for i in range(config['num_experiments']):
        backgrounds = generate_backgrounds(config['sample_size'], config['mean_data'], config['var_data'], image_shape)
        
        for scenario, params in config['parameterizations'].items():
            
            for pattern_scale in params['pattern_scales']:
                
                config['pattern_scale'] = pattern_scale
                
                patterns = scenario_dict.get(scenario)(params=config, image_shape=list(image_shape))
                
                ground_truths = patterns.copy()

                for correlated in ['uncorrelated', 'correlated']:
                    
                    copy_backgrounds = np.zeros((config['sample_size'], image_shape[0] * image_shape[1]))
                    
                    params['correlated_background'] = correlated
                    
                    if correlated == 'correlated':
                        for j, background in enumerate(backgrounds.copy()):
                            copy_backgrounds[j] = gaussian_filter(np.reshape(background, (image_shape[0], image_shape[1])), config['smoothing_sigma']).reshape((image_shape[0] * image_shape[1]))
                        alpha_ind = 1
                    else:
                        copy_backgrounds = backgrounds.copy()
                        alpha_ind = 0

                    for alpha in params['snrs'][alpha_ind]:
                        
                        copy_patterns = patterns.copy()
                        if params['manipulation_type'] == 'multiplicative':
                            copy_patterns = 1 - alpha * copy_patterns

                        normalised_patterns, normalised_backgrounds = normalise_data(copy_patterns, copy_backgrounds)

                        if params['manipulation_type'] == 'multiplicative':
                            x = normalised_patterns * normalised_backgrounds
                        else:
                            x = alpha * normalised_patterns.copy() + (1 - alpha) * normalised_backgrounds.copy()

                        scale = 1 / np.max(np.abs(x))
                        x = np.apply_along_axis(scale_to_bound, 1, x, scale)
                        
                        captured_data[f'experiment_{i}_pattern_plus_background_{scenario}_{correlated}_alpha={alpha}'] = x.copy()
                        
                        # Apply whitening transformation
                        for method in config["whitening_methods"]:
                            whitened_x, _ = whitening_methods[method](x.copy())
                            scale_whitened = 1 / np.max(np.abs(whitened_x))
                            whitened_x = np.apply_along_axis(scale_to_bound, 1, whitened_x, scale_whitened)

                            captured_data[f'experiment_{i}_whitened_{method}_{scenario}_{correlated}_alpha={alpha}'] = whitened_x.copy()

                            scenarios = type(DataScenarios)()
                            class_0_labels = torch.as_tensor([[0]] * int(config['sample_size'] / len(config['patterns'])))
                            class_1_labels = torch.as_tensor([[1]] * int(config['sample_size'] / len(config['patterns'])))

                            whitened_x = torch.as_tensor(whitened_x, dtype=torch.float16)
                            y = torch.ravel(torch.cat((class_0_labels, class_1_labels)))

                            # 90/5/5 split for 64x64 data, 80/10/10 split for 8x8
                            data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=config['test_split'], random_state=SEED)
                            train_indices, val_indices = list(data_splitter.split(X=whitened_x, y=y))[0]

                            x_train = whitened_x[train_indices]
                            y_train = y[train_indices]
                            x_val_test = whitened_x[val_indices]
                            y_val_test = y[val_indices]

                            masks_train = ground_truths[train_indices]
                            masks_val_test = ground_truths[val_indices]

                            data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
                            val_indices, test_indices = list(data_splitter.split(X=x_val_test, y=y_val_test))[0]

                            x_val = x_val_test[val_indices]
                            y_val = y_val_test[val_indices]
                            masks_val = masks_val_test[val_indices]

                            x_test = x_val_test[test_indices]
                            y_test = y_val_test[test_indices]
                            masks_test = masks_val_test[test_indices]

                            correlated_string = "uncorrelated"
                            if correlated == 'imagenet':
                                correlated_string = 'imagenet'
                            elif correlated == 'correlated':
                                correlated_string = "correlated"

                            scenario_key = f'{params["scenario"]}_{config["image_scale"]}d{config["pattern_scale"]}p_{alpha}_{correlated_string}_{method}'

                            scenarios[scenario_key] = DataRecord(x_train, y_train, x_val, y_val, x_test, y_test, masks_train, masks_val, masks_test)

                            dump_as_pickle(data=scenarios, output_dir=output_dir, file_name=scenario_key)
    
    return

def main():
    config_file = 'data_config'
    if len (sys.argv) > 1:
        config_file = sys.argv[1]
    config = load_json_file(file_path=f'data/{config_file}.json') 
    
    date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    folder_path = f'{config["output_dir"]}/{date}'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    whitened_data_generation(config=config, output_dir=folder_path)


if __name__ == '__main__':
    main()
