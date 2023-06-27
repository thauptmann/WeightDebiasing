for METHOD in uniform logistic_regression neural_network_mmd_loss kmm adaDeBoost mrs
do
    python src/weighting_experiment.py --dataset gbs_allensbach --method $METHOD --number_of_repetitions 1
done