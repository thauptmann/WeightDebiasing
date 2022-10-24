python src/run_weighting_experiment.py --dataset census --method none --use_age_bias
python src/run_weighting_experiment.py --dataset census --method none 

python src/run_weighting_experiment.py --dataset census --method logistic_regression --use_age_bias
python src/run_weighting_experiment.py --dataset census --method logistic_regression 

python src/run_weighting_experiment.py --dataset census --method random_forest --use_age_bias
python src/run_weighting_experiment.py --dataset census --method random_forest 

python src/run_weighting_experiment.py --dataset census --method gradient_boosting --use_age_bias
python src/run_weighting_experiment.py --dataset census --method gradient_boosting 

python src/run_weighting_experiment.py --dataset census --method neural_network_classifier --use_age_bias
python src/run_weighting_experiment.py --dataset census --method neural_network_classifier 

python src/run_weighting_experiment.py --dataset census --method neural_network_mmd_loss --use_age_bias
python src/run_weighting_experiment.py --dataset census --method neural_network_mmd_loss 

python src/run_weighting_experiment.py --dataset census --method domain_adaptation --use_age_bias
python src/run_weighting_experiment.py --dataset census --method domain_adaptation 