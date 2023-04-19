DATASET=artificial

python src/run_weighting_experiment.py --dataset $DATASET --method none --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method none --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method none --bias_sample_size 1000

# python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_sample_size 1000

# python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method random_forest --bias_sample_size 1000

# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_classifier --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_classifier --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_classifier --bias_sample_size 1000

# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_sample_size 500 
# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_sample_size 1000 

# python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias_sample_size 1000

# python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_sample_size 100
# python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_sample_size 500
# python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_sample_size 1000