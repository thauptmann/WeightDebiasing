NUMBER_OF_REPETETIONS=3
DROP=10

for BIAS_VARIABLE in random cross-validation temperature class_weights
do
    python src/mrs_ablation_study.py --ablation_experiment $BIAS_VARIABLE --number_of_repetitions $NUMBER_OF_REPETETIONS --drop $DROP
done