python3 extract_features.py \
  --source 'storage/local/BDDA/test/camera_images' \
  --weights yolov5s.pt \
  --conf 0.25  \
  --save-txt \
  --save-conf \
  --features 'storage/local/features/'
