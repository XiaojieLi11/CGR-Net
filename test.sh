cd core/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/DATASET/SIFT/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=True \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/DATASET/SIFT/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=False \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_unknow \
              --data_te=/DATASET/SIFT/yfcc-sift-2000-test.hdf5 \
              --use_ransac=True \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_unknow \
              --data_te=/DATASET/SIFT/yfcc-sift-2000-test.hdf5 \
              --use_ransac=False \
              --log_base=../log/


