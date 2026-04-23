Hi, and welcome to the Entangled Ancillas submission for the 2026 Girls in Quantum qVolution Hackathon, track C - Harmonic Oscillator!

The zipped folder contains:

- jupyter notebook, with our problem statement, analysis and observations
- quantum.py file: this is where our algorithm is stored. It contains 4 main methods: energy_analysis, bound_analysis, k_analysis, n_shots_analysis
  Since each method can take up to a few minutes to run, we have commented them out. If you would like to test them out, we recommend uncommenting and running one a time. The first time the algo is run, you must authenticate (uncomment line at the top of the file)
- plots that will be displayed in the notebook
- and html and pdf of the notebook, in case running locally fails

### How to run locally

1- save and unzip folder in your directory
2- navigate to base folder
3- install and activate venv
4- in your terminal run: `pip install -r requirements.txt`
5- to run the notebook run in your terminal: `jupyter notebook`
6- to test the algorithm, run in your terminal: `python3 quantum.py` (after uncommenting the methods as mentioned earlier)
