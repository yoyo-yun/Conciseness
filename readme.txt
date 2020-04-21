[RA-LSMT model] Conciseness is Better: Recurrent Attention LSTM Model for Document-Level Sentiment Analysis

This PyTorch code was used in the experiments of the research paper

Data
IMDB, Yelp2013, and Yelp2014 datasets are originally from (https://drive.google.com/drive/folders/1PxAkmPLFMnfom46FMMXkHeqIxDbA16oy),
 Amazon are originally from http://jmcauley.ucsd.edu/data/amazon/,
 all datasets are unzip to a directory named corpus as :
-corpus
--IMDB
---file.*
--Yelp_13
---file.*
--Yelp_14
---file.*
--Amazon
---file.*

Train and Evaluate RA-LSTM
    training step (eg. IMDB):
    (1) initializing base model
        python -m train --task IMDB --num_epochs 6 --feature_dim 300 --depth 3 --step base
    (2) initializing ATT-N model on 2nd scale
        python -m train --task IMDB --feature_dim 300 --step att --layer 1
    (3) refining model on 2nd scale
        python -m train --task IMDB --feature_dim 300 --step scale --layer 1 --num_epochs 10
    if no need of sharing same base model, following:
        python -m train --task IMDB --num_epochs 6 --feature_dim 300 --step single
        python -m train --task IMDB --feature_dim 300 --step scale --layer 1 --num_epochs 10 --using_extend 1
    (4) initializing ATT-N model on 3rd scale
        python -m train --task IMDB --feature_dim 300 --step att --layer 2
    (5) refining model on 3rd scale
        python -m train --task IMDB --feature_dim 300 --step scale --layer 2 --num_epochs 10
    if no need of sharing same base model, following:
        python -m train --task IMDB --num_epochs 6 --feature_dim 300 --step single
        python -m train --task IMDB --feature_dim 300 --step scale --layer 2 --num_epochs 20 --using_extend 1
    (6) training fusion classifier
        python -m train --task IMDB --feature_dim 300 --step fusion --depth 3 --num_epochs 10
    (7) evaluating test dataset
        python -m train --task IMDB --step eval