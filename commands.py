# exp0 baseline tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp0 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp1 naive rainbow tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp1 --previewNum 1 --actionType Uniform --num_processes 1  --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp2 naive rainbow IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp2 --previewNum 1 --actionType Uniform --num_processes 1  --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp3 no distributed tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp3 --previewNum 1 --actionType Uniform --num_processes 16 --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp4 no distributed IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp4 --previewNum 1 --actionType Uniform --num_processes 16 --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp5 no para actors tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp5 --previewNum 1 --actionType Uniform --num_processes 1 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp6 no para actors IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp6 --previewNum 1 --actionType Uniform --num_processes 1 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp7 Uniform IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass --custom exp7 --previewNum 1 --actionType Uniform    --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp8 lowest IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass --custom exp8 --previewNum 1 --actionType Uniform    --num_processes 16 --distributed --samplePointsNum 1024  --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp9 HeuAction tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp9 --previewNum 1 --actionType HeuAction  --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp10 HeuAction IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp10 --previewNum 1 --actionType HeuAction  --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp11 LineAction tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp11 --previewNum 1 --actionType LineAction --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp12 LineAction IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp12 --previewNum 1 --actionType LineAction --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp13 RotAction tetris3D_tolerance_middle_mass
# python main.py --disable-cuda --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp13 --previewNum 1 --actionType RotAction  --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

# exp14 RotAction IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp14 --previewNum 1 --actionType RotAction  --num_processes 16 --distributed --samplePointsNum 1024  --resolutionA 0.02 --resolutionH 0.01

#################################################################################################################
# Train with cuda is better.
# exp15 200 action tetris3D_tolerance_middle_mass 6629now
# python main.py --device 3 --data_name tetris3D_tolerance_middle_mass --custom exp15 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 200 --resolutionA 0.02 --resolutionH 0.01
# python main.py --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp15 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 200 --resolutionA 0.02 --resolutionH 0.01

# exp16 200 action IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp16 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 200 --resolutionA 0.02 --resolutionH 0.01
# python main.py  --device 0 --data_name IR_concaveArea3_mass           --custom exp16 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 200 --resolutionA 0.02 --resolutionH 0.01

# exp17 double action resolution tetris3D_tolerance_middle_mass 6629 now
# python main.py --device 2 --data_name tetris3D_tolerance_middle_mass --custom exp17 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.01 --resolutionH 0.01
# python main.py --device 1 --data_name tetris3D_tolerance_middle_mass --custom exp17 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.01 --resolutionH 0.01

# exp18 double action resolution IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp18 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.01 --resolutionH 0.01
# python main.py --device 0 --data_name IR_concaveArea3_mass           --custom exp18 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.01 --resolutionH 0.01

# exp19 double height map resolution tetris3D_tolerance_middle_mass 6629
# python main.py --disable-cuda --device 1 --data_name tetris3D_tolerance_middle_mass --custom exp19 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.005
# python main.py --device 1 --data_name tetris3D_tolerance_middle_mass --custom exp19 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.005

# exp20 double height map resolution IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp20 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.005
# python main.py --device 1 --data_name IR_concaveArea3_mass           --custom exp20 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.005

# exp21 double pointcloud size tetris3D_tolerance_middle_mass 6629 now
# python main.py --device 3 --data_name tetris3D_tolerance_middle_mass --custom exp21 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 2048 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01
# python main.py --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp21 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 2048 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp22 double pointcloud size  IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp22 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 2048 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01
# python main.py --device 0 --data_name IR_concaveArea3_mass           --custom exp22 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 2048 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp23 doubleRot tetris3D_tolerance_middle_mass 6629
# python main.py --disable-cuda --device 1 --data_name tetris3D_tolerance_middle_mass --custom exp23 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01 --doubleRot
# python main.py --device 1 --data_name tetris3D_tolerance_middle_mass --custom exp23 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01 --doubleRot

# exp24 doubleRot IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp24 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01 --doubleRot
# python main.py --device 0 --data_name IR_concaveArea3_mass           --custom exp24 --previewNum 1 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01 --doubleRot

# exp25 Pack it tetris3D_tolerance_middle_mass 6630 to be continue or not
# python main.py --disable-cuda --device 3 --data_name tetris3D_tolerance_middle_mass --custom exp25 --previewNum 10 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01
# python main.py --device 0 --data_name tetris3D_tolerance_middle_mass --custom exp25 --previewNum 10 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# exp26 Pack it IR_concaveArea3_mass
# python main.py --disable-cuda --device 0 --data_name IR_concaveArea3_mass           --custom exp26 --previewNum 10 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01
# python main.py --device 0 --data_name IR_concaveArea3_mass           --custom exp26 --previewNum 10 --actionType Uniform --num_processes 16 --distributed --samplePointsNum 1024 --convexAction Defects --selectedAction 100 --resolutionA 0.02 --resolutionH 0.01

# ssh -p 2004 -L 6624:127.0.0.1:6006 zhaohang@222.244.113.202
# ssh -p 26   -L 6626:127.0.0.1:6006 dell@222.244.113.202
# ssh -p 6629 -L 6629:127.0.0.1:6006 dell@222.244.113.202
# ssh -p 6630 -L 6629:127.0.0.1:6006 zhaohang@222.244.113.202