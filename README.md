# pageReranking
This is the implementation code of pp-hrl on criteo data.

## Data
The constructed data is stored in the `/data/` folder, and the file that uses Gaussian noise to simulate non-click data is stored in `criteo_pv_rec_normalFill.txt`. This is also the final data used in this paper.

## Evaluator
The code of the evaluator used in this paper is stored in the `/simulator/` folder. 

The implementation code of the model is in `model.py`, the loading code of the model is in `load_simulator.py`, and the pre-trained parameters are stored in `simulator.pkl`.

## PP-HRL
We use HMDP to model the page recommendation problem and use the PP-HRL model to solve it. 

Among them, `Env_model.py` describes the modeling of the environment and the HMDP transfer code required for the interaction between the model and the environment. Model implementation details for HLA and LLA are described in `HDDPG_Agent.py`. 

`HDDPG_v2_main.py` is the main code finally used in the experiment. The code can be run by the following command:
```
python HDDPG_v2_main.py
```
