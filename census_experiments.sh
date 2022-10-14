python src/run_weighting_experiment.py --dataset census --method naive --bias none
python src/run_weighting_experiment.py --dataset census --method logistic_regression --bias none
python src/run_weighting_experiment.py --dataset census --method random_forest --bias none
python src/run_weighting_experiment.py --dataset census --method gradient_boosting --bias none
python src/run_weighting_experiment.py --dataset census --method neural_network_classifier --bias none
python src/run_weighting_experiment.py --dataset census --method neural_network_mmd_loss --bias none
python src/run_weighting_experiment.py --dataset census --method domain_adaptation --bias none

python src/run_weighting_experiment.py --dataset census --method naive --bias oversampling
python src/run_weighting_experiment.py --dataset census --method logistic_regression --bias oversampling
python src/run_weighting_experiment.py --dataset census --method random_forest --bias oversampling
python src/run_weighting_experiment.py --dataset census --method gradient_boosting --bias oversampling
python src/run_weighting_experiment.py --dataset census --method neural_network_classifier --bias oversampling
python src/run_weighting_experiment.py --dataset census --method neural_network_mmd_loss --bias oversampling
python src/run_weighting_experiment.py --dataset census --method domain_adaptation --bias oversampling

python src/run_weighting_experiment.py --dataset census --method naive --bias undersampling
python src/run_weighting_experiment.py --dataset census --method logistic_regression --bias undersampling
python src/run_weighting_experiment.py --dataset census --method random_forest --bias undersampling
python src/run_weighting_experiment.py --dataset census --method gradient_boosting --bias undersampling
python src/run_weighting_experiment.py --dataset census --method neural_network_classifier --bias undersampling
python src/run_weighting_experiment.py --dataset census --method neural_network_mmd_loss --bias undersampling
python src/run_weighting_experiment.py --dataset census --method domain_adaptation --bias undersampling