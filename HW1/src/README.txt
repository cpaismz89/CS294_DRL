CS294-112 HW1: Imitation Learning
Author: Cristobal Pais
Date:   September 5th, 2018

Steps for generating the outputs of the report (using the default configuration reported in the pdf file - training epochs = 1000, batch-size = 64, nroll-outs = 20, max-time-steps = 1000:
1) Run 1_Experts.bat (Windows batch file) to generate all the experts' data and csv files containing relevant information to plot 
2) Run 2_BC.bat to generate all the outputs from our initial behavioral cloning algorithm (3 NN model)
3) Run 3_HyperEpochs.bat to generate all the outputs for the hyper-parameter tuning performance comparison (number of epochs using the Humanoid task)
4) Run 4_DAgger.bat to generate all the outputs from the DAgger algorithm (10 iterations by default)
5) Run 5_DaggerNewNN.bat to run the modified models (extra layers/neurons and different activation functions models) and generate the relevant outputs for the Humanoid task 
6) Run 6_Plots.bat to generate all the plots and LaTeX tables included in the report
7) Run 6_PlotsSample.bat to generate the plots included in the report (specific random seed) based on the provided outputs inside the SampleOutputs folder

Notes:
- Render is disabled by default 
- All trained models are included for reference inside the Models Folder. These are generated from scratch when running the .bat files, as requested.
- To modify any parameter of the algorithms, edit the .bat files 
- For the modified model, the number of layers/neurons and activation functions can be modified via command line arguments (disabled in the submitted version for simplicity). Refer to lines 109-119 inside the run_DAgger_Free.py file for modifying the network structure inside the code.
- Thanks for giving us an interesting, useful, and entertaining first HW!
- All Figures and Tables are included inside the report HW1_CS294_CPais.pdf. Description of the results and models applied are included in Figures and Tables captions, as required.
- An Appendix section is included in the report for an easy visualization of the plots and tables obtained.