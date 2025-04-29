import os
import shutil

# Split data into V / NV format instead of based on csv
move_dirs = ['test', 'train', 'valid']

for dir in move_dirs:
    source_folder = f'data2/{dir}'
    v_folder = f'data2/{dir}/V'
    nv_folder = f'data2/{dir}/NV'

    os.makedirs(v_folder, exist_ok=True)
    os.makedirs(nv_folder, exist_ok=True)


    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg')):
            first_word = filename.split('_')[0]
            src_path = os.path.join(source_folder, filename)

            if first_word == 'V':
                dest_path = os.path.join(v_folder, filename)
            elif first_word == 'NV':
                dest_path = os.path.join(nv_folder, filename)
            else:
                continue

            shutil.move(src_path, dest_path)
