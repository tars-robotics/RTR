# end_frame trims an extra 5 frames; vase_sponge_test1 trims 20 because its final stage lifts up and end_frame is judged from the x-axis, so the extra trim is needed.
python scripts/process_data/peel_cucumber/process_data_all_zarr_60hz.py \
    --root_path ./data/raw \
    --save_path ./data/processed_peel_huanggua/peel_huanggua_60hz \
    --task_list peel_huanggua0107 \
    --episode_length 80
    # --save_camera_vis \
    # --episode_length 2

