NUMBER_OF_REPETETIONS=100
BIAS_TYPE=mean_difference

for METHOD in uniform logistic_regression kmm soft-mrs soft-mrs-exponential mrs neural_network_mmd_loss
do
    for DATASET in breast_cancer loan_prediction hr_analytics  folktables_income folktables_employment 
    do
        python src/weighting_experiment.py --dataset $DATASET --method $METHOD \
        --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS  
    done
done