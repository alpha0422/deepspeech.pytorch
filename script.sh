#!/usr/bin/bash

# Install own driver
#sudo /home/scratch.computelab/utils/driver/install_driver.py --installer /home/builds/daily/display/x86_64/dev/gpu_drv/cuda_a/20171201_23218477/NVIDIA-Linux-x86_64-dev_gpu_drv_cuda_a-20171201_23218477.run --reason="need driver for CUDA 9.1"
#install_driver

# Run Deep Speech 2
#python train.py --train_manifest data/libri_train_manifest_f20kl.csv --val_manifest data/libri_val_manifest.csv --hidden_size 2560 --no_bidirectional --epochs 1 --batch_size 32 --cuda --fp16
#python train.py --train_manifest data/dummy-4s/dummy.csv --val_manifest data/an4_val_manifest.csv --conv_layers 2 --conv_type Conv2d --hidden_layers 5 --hidden_size 2560 --rnn_type gru --no_bidirectional --fc_layers 2 --bn --epochs 1 --batch_size 128 --cuda --fp16
#nvprof --profile-child-processes --profile-from-start off -o nvprof.%p.export --log-file nvprof.%p.log --csv --print-gpu-trace python train.py --train_manifest data/dummy-4.53s/dummy.csv --val_manifest data/an4_val_manifest.csv --conv_layers 2 --conv_type Conv2d --hidden_layers 5 --hidden_size 2560 --rnn_type gru --no_bidirectional --fc_layers 2 --bn --epochs 1 --batch_size 128 --cuda --fp16 --nvprof

# Run nvprof for nvvp
for size in 128
do
    nvprof --profile-child-processes --profile-from-start off -o nvprof.%p.export --log-file nvprof.%p.log --csv --print-gpu-trace python train.py --train_manifest data/libri_train_manifest_f20kl.csv --val_manifest data/libri_val_manifest.csv --hidden_size 2560 --no_bidirectional --epochs 1 --batch_size ${size} --cuda --fp16 --nvprof | tee log.txt &
    #nvprof --profile-child-processes --profile-from-start off -o nvprof.%p.export --log-file nvprof.%p.log --csv --print-gpu-trace python train.py --train_manifest data/dummy/dummy.csv --val_manifest data/an4_val_manifest.csv --hidden_size 2048 --no_bidirectional --epochs 1 --batch_size 128 --cuda --fp16 --nvprof | tee log.txt &
    #nvprof --profile-child-processes --profile-from-start off -o nvprof.%p.export --log-file nvprof.%p.log --csv --print-gpu-trace python train.py --train_manifest data/dummy1/dummy.csv --val_manifest data/an4_val_manifest.csv --hidden_size 2560 --no_bidirectional --epochs 1 --batch_size 128 --cuda --fp16 --nvprof | tee log.txt &
    sleep 10
    pid=$(pgrep python -U wkong -n)
    wait $!
    python plot_nvprof.py --in_files nvprof.${pid}.export --out_file nvprof.${pid}.txt
    python mark_layer.py --log_file nvprof.${pid}.log --plot_file nvprof.${pid}.xlsx --out_file nvprof.${pid}.report.xlsx
    mv log.txt nvprof.${pid}.run.log
    zip -r nvprof.${pid}.zip nvprof.${pid}.*
    find . -maxdepth 1 -name "nvprof.${pid}.*" -and ! -name '*.zip' -delete
done

# Run nvprof metrics
#nvprof --profile-child-processes --profile-from-start off --kernels volta_sgemm_128x64_nn --metrics achieved_occupancy,flop_count_sp,flop_count_sp_fma,flop_sp_efficiency --csv python train.py --train_manifest data/an4_train_manifest.csv --val_manifest data/an4_val_manifest.csv --epochs 1 --batch_size 64 --cuda --fp16 --nvprof
#nvprof --profile-child-processes --profile-from-start off --kernels volta_scudnn_128x32_relu_interior_nn_v1 --metrics achieved_occupancy,flop_count_sp,flop_count_sp_fma,flop_sp_efficiency --csv python train.py --train_manifest data/an4_train_manifest.csv --val_manifest data/an4_val_manifest.csv --epochs 1 --batch_size 64 --cuda --fp16 --nvprof

