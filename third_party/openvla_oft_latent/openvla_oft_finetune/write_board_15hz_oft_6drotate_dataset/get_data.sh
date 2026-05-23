ZARR_PATH="../../../../data/ckpts/write_board_15hz/rdp_zarr/replay_buffer.zarr"
OUTPUT_DIR="./data"

python -m rtr_async_sys.tools.convert_zarr_into_episodes_for_openvla_oft_6d_rotate \
    --zarr_path "${ZARR_PATH}" \
    --output_dir "${OUTPUT_DIR}"

# then
tfds build --overwrite
