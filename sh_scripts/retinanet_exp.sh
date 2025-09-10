#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task angle -bs 2 -concept_source retinanet -train_concepts -seed 42
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task distance -bs 2 -concept_source retinanet -train_concepts -seed 42
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task multitask -bs 2 -concept_source retinanet -train_concepts -seed 42
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task angle -bs 2 -concept_source retinanet -train_concepts -seed 123
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task distance -bs 2 -concept_source retinanet -train_concepts -seed 123
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task multitask -bs 2 -concept_source retinanet -train_concepts -seed 123
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task angle -bs 2 -concept_source retinanet -train_concepts -seed 2025
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task distance -bs 2 -concept_source retinanet -train_concepts -seed 2025
#TRAIN
!python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 100 -task multitask -bs 2 -concept_source retinanet -train_concepts -seed 2025
