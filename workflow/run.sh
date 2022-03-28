#! /bin/bash    
../scripts/build_gen1d.py \
    --json_gz ../data/ligands_erd3_sample.json.gz \
    --output p/vocab_erd3.json.gz

../scripts/build_gen1d.py \
    --json_gz ../data/ligands_pdbl.json.gz \
    --output p/vocab_pdbl.json.gz

../scripts/build_gen2d.py \
    --structures ../data/ligands_erd3_sample.json.gz \
    --vocabulary p/vocab_erd3.json.gz \
    --output q/model_q_erd3.arch

../scripts/build_gen2d.py \
    --input q/model_q_erd3.arch \
    --structures ../data/ligands_pdbl.json.gz \
    --vocabulary p/vocab_pdbl.json.gz \
    --recalibrate \
    --epochs 10 \
    --output q/model_q_pdbl.arch 

cd ../data && tar -xf complexes_sample.tar.gz && cd ../workflow
../scripts/build_gen3d.py \
    --complexes ../data/complexes_sample.json \
    --complexes_skiplist ../data/complexes_codes_testset.json \
    --vocabulary p/vocab_pdbl.json.gz \
    --baseline q/model_q_pdbl.arch \
    --output r/model_r.arch 

