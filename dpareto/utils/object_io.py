# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import os


def save_object(output_dir, obj, name, save_pickle=True, save_text=True):
    os.makedirs(output_dir, exist_ok=True)

    if save_pickle:
        # Save pickle of object
        with open('{}/{}.pkl'.format(output_dir, name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    if save_text:
        # Save text of object
        with open('{}/{}.txt'.format(output_dir, name), 'w') as f:
            if type(obj) in (list, tuple):
                obj_string = ''
                for item in obj:
                    obj_string += str(item) + '\n'
            elif type(obj) is dict:
                obj_string = ''
                for key, value in obj.items():
                    obj_string += str(key) + '\n - ' + str(value) + '\n'
            else:
                obj_string = str(obj)
            f.write(obj_string)
