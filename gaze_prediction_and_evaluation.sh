python3 gaze_prediction_and_evaluation.py \
        --data 'storage/local/features'  \
        -b 64 \
        --lr 0.01 \
        --epochs 40 \
        --no_train \
        --gridheight 16 \
        --gridwidth 16 \
        --traingrid 'storage/local/grids/grid1616/training_grid.txt' \
        --valgrid 'storage/local/grids/grid1616/validation_grid.txt' \
        --testgrid 'storage/local/grids/grid1616/test_grid.txt' \
        --best  'storage/local/grid1616_model_best.pth.tar' \
        --gazemaps 'storage/local/BDDA/test/gazemap_images' \
        --yolo5bb 'storage/local/yolo5_boundingboxes/' \
        --visualizations 'storage/local/results/grid1616/' \
        --threshhold 0.5 #\
        #--lstm \   
        #--convlstm \
        #--sequence 8
