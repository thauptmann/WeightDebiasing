python src/run_weighting_experiment.py --dataset barometer --method none --use_age_bias --bias_sample_size 1000
python src/run_weighting_experiment.py --dataset barometer --method none --bias_sample_size 1000

python src/run_weighting_experiment.py --dataset barometer --method logistic_regression --use_age_bias --bias_sample_size 1000
python src/run_weighting_experiment.py --dataset barometer --method logistic_regression --bias_sample_size 1000

python src/run_weighting_experiment.py --dataset barometer --method random_forest --use_age_bias --bias_sample_size 1000
python src/run_weighting_experiment.py --dataset barometer --method random_forest --bias_sample_size 1000 

python src/run_weighting_experiment.py --dataset barometer --method neural_network_classifier --use_age_bias --bias_sample_size 1000
python src/run_weighting_experiment.py --dataset barometer --method neural_network_classifier --bias_sample_size 1000 

python src/run_weighting_experiment.py --dataset barometer --method neural_network_mmd_loss --use_age_bias --bias_sample_size 1000
python src/run_weighting_experiment.py --dataset barometer --method neural_network_mmd_loss --bias_sample_size 1000

python src/run_weighting_experiment.py --dataset barometer --method domain_adaptation --use_age_bias
python src/run_weighting_experiment.py --dataset barometer --method domain_adaptation 

python src/run_weighting_experiment.py --dataset barometer --method random --use_age_bias
python src/run_weighting_experiment.py --dataset barometer --method random 