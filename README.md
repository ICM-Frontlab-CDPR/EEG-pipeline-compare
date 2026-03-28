### **General Description**

This python package is design to compare derivatives/ of EEG pipelines, based on standardize metrics and type of analysis.

It aims to help the standarization of EEG preprocessing steps in analyzing where pipelines:

- diverge or not (fragmentation check)
- for good reason or not (quality check)



Possible use case :

- Comparison of pipelines
- Comparison of hardware devices

### **Quick start**

0- Run the pipelines that you want to compare and give the list of derivatives path.

0bis - To be treated, derivatives/ (Pipelines output) has to declare a derivatives/ structure. (otherwise it will be a file comparison) Default is BIDS (to check with bids-validator ? could be integrated)

**1 - Define metrics of comparison**

- canal rejections
- ICA components
- statistical properties of the signal

**2- Define at what stage of the preprocess the comparison is made**

- before epochs
- after epochs
- features extractions files

### **Features**

**1- Integration of a mapping between objectives and metrics**

- take a standardize objective
- 

### **Outputs**

2 types of outputs comparison matrix :

- Dual derivatives/ comparison : Visualisation of multiple metrics
- Multiple pipeline comparison : Visualisation of one metrics

**---- Long-term approch :**

1- Be clear about what we call an EEG preprocessing pipeline :

- if pythonic, must have a pyprojects.toml
- have to declare his type of analysis span

2- Registering the existing one

3- Dependencies graph tools : The general idea is the following : when desiging an EEG pipeline, if you have the same dependencies of another registered pipeline you are probably creating something new for nothing.

++++++++ later

 a tool adding an EEG decorator to pipelines : this decorator has to condition every function used by type of analysis, type of hardware, etc.
