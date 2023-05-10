DATASET=breast_cancer

for BIAS_VARIABLE in clump_thickness uniformity_of_cell_size uniformity_of_cell_shape marginal_adhesion single_epithelial_cell_size \
bare_nuclei bland_chromatin normal_nucleoli mitoses class none
do
    python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type none --bias_variable "$BIAS_VARIABLE"
    python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type none --bias_variable "$BIAS_VARIABLE"
    python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type none --bias_variable "$BIAS_VARIABLE"
    python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type none --bias_variable "$BIAS_VARIABLE"
    python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type none --bias_variable "$BIAS_VARIABLE"
    # python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type none --bias_variable "$BIAS_VARIABLE"
done

python src/run_weighting_experiment.py --dataset $DATASET --method uniform --bias_type mean_difference --bias_variable none
python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type mean_difference --bias_variable none
python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type mean_difference --bias_variable none
python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type mean_difference --bias_variable none
python src/run_weighting_experiment.py --dataset $DATASET --method adaDeBoost --bias_type mean_difference --bias_variable none
# python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type mean_difference --bias_variable none