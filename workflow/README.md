## Workflow

Note that the [../data](../data) directory contains only a small data sample. With the commands below, you can thus verify the successful setup of the python environment and sound code execution. To train performant models, you will first need to download the complete datasets (see [these](../data/README.md) instructions).


### 1. Construct the pseudo-generative 1D model (G<sub>p</sub>)
```bash
../scripts/build_gen1d.py \
    --json_gz ../data/ligands_erd3_sample.json.gz \
    --output p/vocab_erd3.json.gz
../scripts/build_gen1d.py \
    --json_gz ../data/ligands_pdbl.json.gz \
    --output p/vocab_pdbl.json.gz
```
### 2. Train the 2D model (G<sub>pq</sub>)
```bash
../scripts/build_gen2d.py \
    --structures ../data/ligands_erd3_sample.json.gz \
    --vocabulary p/vocab_erd3.json \
    --output q/model_q_erd3.arch
```
### 3. Recalibrate the 2D model (G'<sub>pq</sub>)
```bash
../scripts/build_gen2d.py \
    --input q/model_erd3.arch \
    --structures ../data/ligands_pdbl.json.gz \
    --vocabulary p/vocab_pdbl.json.gz \
    --recalibrate \
    --epochs 10 \
    --output q/model_q_pdbl.arch 
```
### 4. Train the 3D model (G<sub>pqr</sub>)
```bash
# Unpack data sample (demo only)
cd ../data && tar -xf complexes_sample.tar.gz && cd ../workflow
../scripts/build_gen3d.py \
    --complexes ../data/complexes_sample.json \
    --complexes_skiplist ../data/complexes_codes_testset.json \
    --vocabulary p/vocab_pdbl.json.gz \
    --baseline q/model_pdbl.arch \
    --output r/model_r.arch
```
### 5. Prospective sampling
```bash
../scripts/sample_spt.py \
    --complexes ../data/complexes_sar_examples.json \
    --vocabulary p/vocab_pdbl.json.gz \
    --baseline q/model_q_pdbl.arch \
    --model r/model_r.arch \
    --verbose
```
