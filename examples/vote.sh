for solve in True False
do
    python /home/chi/NNRobustness/ensembleKW/examples/voting.py \
        --model_type mnist_small_0_1 \
        --root /home/chi/NNRobustness/ensembleKW/evalData/l_inf/ \
        --count 7 \
        --weights None \
        --solve_for_weights $solve \

    python /home/chi/NNRobustness/ensembleKW/examples/voting.py \
        --model_type mnist_small_0_3 \
        --root /home/chi/NNRobustness/ensembleKW/evalData/l_inf/ \
        --count 3 \
        --weights None \
        --solve_for_weights True \

    # python /home/chi/NNRobustness/ensembleKW/examples/voting.py \
    #     --model_type mnist_small_0_3 \
    #     --root /home/chi/NNRobustness/ensembleKW/evalData/l_inf/ \
    #     --count 3 \
    #     --weights None \
    #     --solve_for_weights False \


done