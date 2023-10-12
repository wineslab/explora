## Experiments

### Organization

The folder contains 3 sub-folders:
- [`offline-training/`](offline-training) contains the offline datasets and agents' weights (both agents `eMBB` - `HT` in the paper - and `URLLC` - `LL` in the paper)  that were kindly provided by the authors of 
  > [PBO+22] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control on Programmable Experimental Platforms,” IEEE Transactions on Mobile Computing, July 2022.
- [`motivation_main-results/`](motivation_main-results) contains the experiments for Section 3, Section 6.2 and Appendix C of our paper.
- [`action-steering/`](action-steering) contains the experiments for the Section 6.3 and Appendix D of our paper.

### Agents, Slices and Traffic Patterns

We run experiments for three different slices similarly to [PBO+22], namely *eMBB*, *MTC*, *URLLC* with two different agents, denoted as `eMBB` - `HT` in the paper - and `URLLC` - `LL`. At the time of resource allocation, the agent `embb` gives preference to the slice *eMBB* while the agent `urllc` gives preference to the slice *URLLC*. We also use two different traffic patterns, hereafter we summarize their properties, that are used to configure [`MGEN`](https://github.com/USNavalResearchLaboratory/mgen/blob/master/doc/mgen.pdf).

Traffic pattern 1 (trf1):
- eMBB: `5,595,UDP,PERIODIC,[357.15 1400]`
- MTC: `5,595,UDP,POISSON,[44.64 125]`
- URLLC: `5,595,UDP,POISSON,[89.29 125]`

Traffic pattern 2 (trf2):
- eMBB: `5,3595,UDP,PERIODIC,[357.15 700]`
- MTC: `5,3595,UDP,POISSON,[133.92 125]`
- URLLC: `5,3595,UDP,POISSON,[178.58 125]`

### Detailed Configuration

#### Experiments for the [`motivation_main-results/`](motivation_main-results)

All the experiments run on Colosseum/SCOPE for 20 minutes.

`embb-trf1`:

| UE | Exp Number | eMBB UE | MTC UE | URLLC UE |
|:--:|:----------:|:-------:|:------:|:--------:|
| 6 | 1 | 2 | 2 | 2 |
| 5 | 2 | 2 | 1 | 2 |
| 4 | 3 | 1 | 1 | 2 |
| 3 | 4 | 1 | 1 | 1 |
| 2 | 5 | 1 | 0 | 1 |
| 1 | 6 | 0 | 0 | 1 |
| 1 | 7 | 1 | 0 | 0 |
| 1 | 8 | 0 | 1 | 0 |

`embb-trf2`:

| UE | Exp Number | eMBB UE | MTC UE | URLLC UE |
|:--:|:----------:|:-------:|:------:|:--------:|
| 6  | 9 | 2 | 2 | 2 |
| 5  | 10 | 2 | 1 | 2 |
| 4  | 11 | 1 | 1 | 2 |
| 3  | 12 | 1 | 1 | 1 |
| 2  | 13 | 1 | 0 | 1 |
| 1  | 14 | 0 | 0 | 1 |
| 1  | 15 | 1 | 0 | 0 |
| 1  | 16 | 0 | 1 | 0 |

`urllc-trf1`

| UE | Exp Number | eMBB UE | MTC UE | URLLC UE |
|:--:|:----------:|:-------:|:------:|:--------:|
| 6  | 27 | 2 | 2 | 2 |
| 5  | 28 | 2 | 1 | 2 |
| 4  | 29 | 1 | 1 | 2 |
| 3  | 30 | 1 | 1 | 1 |
| 2  | 31 | 1 | 0 | 1 |
| 1  | 32 | 0 | 0 | 1 |
| 1  | 33 | 1 | 0 | 0 |
| 1  | 34 | 0 | 1 | 0 |

`urllc-trf2`

| UE | Exp Number | eMBB UE | MTC UE | URLLC UE |
|:--:|:----------:|:-------:|:------:|:--------:|
| 6  | 35 | 2 | 2 | 2 |
| 5  | 36 | 2 | 1 | 2 |
| 4  | 37 | 1 | 1 | 2 |
| 3  | 38 | 1 | 1 | 1 |
| 2  | 39 | 1 | 0 | 1 |
| 1  | 40 | 0 | 0 | 1 |
| 1  | 41 | 1 | 0 | 0 |
| 1  | 42 | 0 | 1 | 0 |

#### Experiments for the [`action-steering/`](action-steering) results

These results are obtained processing experiments that are slightly different than the above ones. First, the number of users changes during the course of the experiment with a drop from 6 (2 users per slice) to 5 to emulate changing conditions. Then, the experiments follow the next workflow
1. network runs for a specific amount of time (10 minutes) - *phase-a*
2. 1 eMBB user drops at minute 5 of *phase-a*
3. an online training phase takes place (10-15 minutes) to make the corresponding agent aware of the changes; in this phase the agent explores states, hence could take non-optimal decisions - *phase-b*
4. the network resumes usual operations (5-10 minutes) - *phase-c*; during this phase we implement the *action replacement* (*AR* in the paper, see Section 5.2) *strategies*.

We test two differnt observation windows (*O* in the paper):
- `10`, with experiment configuration (*Note: the results in the paper are derived using this configuration*): 
  - *phase-a*: 10 minutes, 
  - *phase-b* 15 minutes,
  - *phase-c* 5 minutes.
- `20`: with experiment configuration: 
  - *phase-a*: 10 minutes, 
  - *phase-b* 10 minutes, 
  - *phase-c* 10 minutes.
