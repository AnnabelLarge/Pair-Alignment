ABOUT: compare my implementation of the GTR substitution model to XRATE

STATUS: PASS [8/8/25]
	max relative difference: 1.0767091825450945e-05




setup:
- one GTR, unnormalized
- unique branch length per cherry pair
- no indel model, but there IS a score for geometric sequence lengths

experiment:
1.) fit joint loglikelihoods with XRATE
2.) convert XRATE parameters to numpy arrays compatible with my model
3.) load XRATE parameters, evaluate with my code
4.) compare loglikes in bits and nats
    > note: XRATE is numerically less precise

result files:
natscore_difference.tsv
bitscore_difference.tsv
