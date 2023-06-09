NUMBER_OF_REPETETIONS=5

for DATASET in folktables_income folktables_employment breast_cancer hr_analytics loan_prediction
do
    # for METHOD in uniform logistic_regression neural_network_mmd_loss kmm adaDeBoost mrs
    for METHOD in adaDeBoost
    do
        for BIAS_TYPE in less_positive_class less_negative_class mean_difference none
        do
            python src/weighting_experiment.py --dataset $DATASET --method $METHOD \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS
        done
    done
done