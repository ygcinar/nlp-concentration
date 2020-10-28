# nlp-concentration

# Original source code for the Word Representations Concentrate and This is Good News! CONNL 2020 

to print arguments for training and evaluation:

`python train_evaluate.py -h`

to train (and evaluate) lssvm with RBF kernel (sigma2=2) using 500 samples of yahoo answers dataset "Society & Culture" vs. "Education & Reference" categories using glove embeddings:
`python train_evaluate.py -f lssvm -n 500  -d yqa -e dc-tfidf-glv -c "Society & Culture" "Education & Reference" -k rbf -r 2.0 -s 2.0`

to train (and evaluate) lssvm with RBF kernel (sigma2=2) using 2000 samples of yahoo answers dataset "Society & Culture" vs. "Education & Reference" categories using tf-idf features:
`python train_evaluate.py -f lssvm -n 2000  -d yqa -e dc-tfidf -c "Society & Culture" "Education & Reference" -k rbf -r 2.0 -s 2.0`

to train (and evaluate) lssvm with polynomial kernel (f(t)=10 f'(t)=-5 f''(t)=1) using 2000 samples of yahoo answers dataset "Society & Culture" vs. "Education & Reference" categories using word2vec embeddings:
`python train_evaluate.py -f lssvm -n 2000  -d yqa -e dc-tfidf-w2v -c "Society & Culture" "Education & Reference" -k poly -t 10 -5 1`

Scripts for some figures are given under scripts/
example of running scripts for Figure 2:  

`cd scripts`  
`./run_fig2.txt`
