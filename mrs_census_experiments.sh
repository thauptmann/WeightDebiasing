DATASET=mrs_census
BIAS_VARIABLE=Above_Below\ 50K

# python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type less_negative_class --bias_variable "$BIAS_VARIABLE"
python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type high_bias --bias_variable "$BIAS_VARIABLE"

# python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type less_negative_class --bias_variable "$BIAS_VARIABLE"
python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type high_bias --bias_variable "$BIAS_VARIABLE"

# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type less_negative_class --bias_variable "$BIAS_VARIABLE"
python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type high_bias --bias_variable "$BIAS_VARIABLE"

# python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type less_negative_class --bias_variable "$BIAS_VARIABLE"
python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type high_bias --bias_variable "$BIAS_VARIABLE"

# python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type less_negative_class --bias_variable "$BIAS_VARIABLE"
python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type high_bias --bias_variable "$BIAS_VARIABLE"

#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type less_positive_class --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type mean_difference --bias_variable "$BIAS_VARIABLE"