"clump_thickness",
"uniformity_of_cell_size",
"uniformity_of_cell_shape",
"marginal_adhesion",
"single_epithelial_cell_size",
"bare_nuclei",
"bland_chromatin",
"normal_nucleoli",
"mitoses",
"class",

DATASET=breast_cancer

#python src/run_weighting_experiment.py --dataset $DATASET --method none --bias_type high_bias --bias_variable "$BIAS_VARIABLE"
# python src/run_weighting_experiment.py --dataset $DATASET --method logistic_regression --bias_type none --bias_variable "$BIAS_VARIABLE"
# python src/run_weighting_experiment.py --dataset $DATASET --method neural_network_mmd_loss --bias_type none --bias_variable "$BIAS_VARIABLE"
# python src/run_weighting_experiment.py --dataset $DATASET --method kmm --bias_type none --bias_variable "$BIAS_VARIABLE"
# python src/run_weighting_experiment.py --dataset $DATASET --method adaDebias --bias_type none --bias_variable "$BIAS_VARIABLE"
#python src/run_weighting_experiment.py --dataset $DATASET --method mrs --bias_type none --bias_variable "$BIAS_VARIABLE"