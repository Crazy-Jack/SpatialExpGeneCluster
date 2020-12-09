python ../src/trainMyPCL.py --dataset spatial \
--experiment_name pcl_cnn \
--model resnet50 \
--batch_size 64 \
--lr_scheduling warmup \
--optimizer SGD \
--learning_rate 0.1 \
--data_root ../data/spatial_Exp/