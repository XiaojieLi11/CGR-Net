
cd core/

python main.py --run_mode=train      \
              --log_base=../log/     \
              --data_tr=/DATASET/SIFT/yfcc-sift-2000-train.hdf5 \
              --data_va=/DATASET/SIFT/yfcc-sift-2000-val.hdf5

