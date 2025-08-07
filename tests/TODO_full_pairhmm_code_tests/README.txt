ABOUT: With just one fragment type, both indp-sites and frag-and-site-mix models reduce to the same model (with the same sample loglikes, same parameters). Overtrain on a small subset of example alignments to prove this

(technically already tested at tests/pairhmm_fragment_and_site_class_mixtures/frag_mix_reduction_test.py; this tests EVERYTHING at the full-code level)


probably train with indp sites code, then eval with frag mix code



MODELS TO TEST:
  - f81 + tkf92
  - gtr + tkf92


FOLDER CONTENTS:
================

config files (run these to recreate results):
----------------------------------------------
CONFIG_frag-mix-code_f81_tkf92.json
CONFIG_frag-mix-code_gtr_tkf92.json
CONFIG_indp-sites-code_f81_tkf92.json
CONFIG_indp-sites-code_gtr_tkf92.json

