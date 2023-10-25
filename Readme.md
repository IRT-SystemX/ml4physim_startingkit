# Starting Kit - Machine Learning for Physical Simulation Challenge
The starting kit provides a set jupyter notebooks and scripts helping the challenge participants to better understand the use case, data, and how actually cotribute in this compeition. For general information concerning the challenge and submit your solutions, you can refere to the competition [Codabench page](https://www.codabench.org/competitions/1534/).

In the following, we describe the content of this repository. We start by describing the jupyter notebooks (these notebook are ordered to facilitate the comprehension of the challenge and contribution of the participants):

- **0_Basic_Competition_Information**: This notebook contains the general information concerning the competition organisation, phases, deadlines and terms. The content is same as the one shared in the competition Codabench page. It should be carefully readen before any contribution and submission. *The participation of Non legal participants will be ignored.* 

- **1-Airfoil_design_basic_simulation**: This notebook aims to familiarize the participants with the use case and to facilitate their comprehension. It shows how to manipulate the Airfoil usecase through some visualizations.

- **2-Import_Airfoil_design_Dataset**: Shows how the challenge datasets could be downloaded and imported using proper functions. These data will be used in the following notebook to train and evaluate an augmented simulator. 

- **3-Reproduce_baseline_results**: This notebook shows how the baseline results shared with participants couldbe reproduced. It include the whole pipleline of training, evaluation and scoring of an augmented simulator using [LIPS platform](https://github.com/IRT-SystemX/LIPS).

- **4-How_to_Contribute**: This notebook shows 3 ways of contribution for begginer, intermediate and advanced users. The submissions should respect one of these forms to be valid and also to enable their proper evaluation through the LIPS platform which will be used for the final evaluation of the results.

- **5-Scoring**: This notebook shows firslty how the score is computed by describing its different components. Next, it provides a script which can be used locally by the participants to obtain a score for their contributions. **We encourage strongly the participants to evaluate locally their solutions and obtain a score, before their submission on Codabench.** It will avoid the submission of incorrect solutions and also save resources to allocate for the needs of other participants.

- **6_Submit_to_Codabench**: It shows, step-by-step, how to submit your solution (augmented simulator) using the [competition Codabench page](https://www.codabench.org/competitions/1534/). 

Other folder and files in this starting kit are :

- **configs folder**: It includes configuration files required whether by a physical solver for importing the datasets or by an augmented simulator to import the set of its hyperparameters. 

- **input_data_local**: It contains few observations (demo dataset) to verify that all the configuration works properly and the participants could train and evaluate a simple Augmented Simulator without any errors.

- **related_papers**: It includes the scientific papers related to this competition. We encourage the participants to read these papers for more details and information concerning the evaluation framework and the considered use case.

