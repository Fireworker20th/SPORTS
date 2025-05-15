#bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep1.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_sigmoid_stride2_mask_embed_link_ffn_joint_train_viper_base12/latest.pth
#echo "Waiting for 3 minutes..."
#sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep2.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep1/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
for train in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
do
   bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep2.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep2/latest.pth
   echo "Waiting for 3 minutes..."
   sleep 180 # 等待300秒，‌即五分钟
   echo "The current date and time is: $train"
done

bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep19.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep2/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep19.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep19/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep21.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep19/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep21.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep21/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep23.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep21/latest.pth
echo "Waiting for 3 minutes..."
sleep 180 # 等待300秒，‌即五分钟
bash ./tools/dist_train.sh configs/det/video_knet_viper/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep23.py 1 $WORK_DIR --no-validate --load-from work_dirs/video_knet_s3_r50_rpn_1x_vkitti2_fusion_sigmoid_stride2_mask_embed_link_ffn_joint_train_ep23/latest.pth