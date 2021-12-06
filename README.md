Members - Keshav Sapkota, Nasir Said, Xin Shu
06/12/2021

Main data source file - https://data.smartdublin.ie/dataset/dublinbikes-api/resource/1c18f219-3885-403e-aa55-4d4c78ee0204

Each file is a stand-alone file. All the models are separated in single 
file so they can be run separately. 

Station 33 was analysed in the report which can be found in the dataset 
folder. This file already comes with labelled classes which was achieved 
by running data_labelling.py on raw file obtained from above link. 

To run the models, simply open the desired model and run it. All the 
related parameters, and graphs will be generated. 

evaluation.py was used to bring all the models together to print a single ROC curve for 
performance comparison. 
