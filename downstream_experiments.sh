NUMBER_OF_REPETETIONS=50

for METHOD in uniform logistic_regression kmm soft-mrs mrs
do
    for DATASET in breast_cancer loan_prediction hr_analytics folktables_income folktables_employment 
    do
        for BIAS_TYPE in less_negative_class less_positive_class mean_difference
            do
            python src/weighting_experiment.py --dataset $DATASET --method $METHOD \
            --bias_type $BIAS_TYPE --number_of_repetitions $NUMBER_OF_REPETETIONS  
            done
    done
done