DATASET=mrs_census

python src/run_weighting_experiment.py --dataset $DATASET --method none --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method none --bias less_positive_class
python src/run_weighting_experiment.py --dataset $DATASET --method none --bias less_negative_class

python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias less_positive_class
python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias less_negative_class

python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias less_positive_class
python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias less_negative_class


python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias less_positive_class 
python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias less_negative_class


python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias less_positive_class 
python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias less_negative_class 

python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias none
python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias less_positive_class
python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias less_negative_class

#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias none
#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias less_positive_class
#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias less_negative_class