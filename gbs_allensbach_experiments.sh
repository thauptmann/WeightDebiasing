
python src/run_weighting_experiment.py --dataset gbs_allensbach --method none 
python src/run_weighting_experiment.py --dataset gbs_allensbach --method logistic_regression
python src/run_weighting_experiment.py --dataset gbs_allensbach --method neural_network_mmd_loss 
python src/run_weighting_experiment.py --dataset gbs_allensbach --method kmm 
python src/run_weighting_experiment.py --dataset gbs_allensbach --method adaDebias 
# python src/run_weighting_experiment.py --dataset gbs_allensbach --method mrs 