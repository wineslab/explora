"""
Plot SHAP computing times

10/13/2023
_______________
    
Summary: plotting for two configuration settings (embb_trf2 and urllc_trf2) all the computing times on the RTX3090
         This is the plot in figure 4(b) of the manuscript.

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib

#^ set configurations  
X = ['C1','C2','C3','C4','C5','C6']
#^ computing times taken from /results/shap_explanations/processing_times/gpu-rtx_3090
embb_trf2 = [58524.97863173485,59219.27416014671,3407.0903618335724, 719.325023651123, 712.5634872913361,677.6995697021484]
urllc_trf2 = [60043.459069252014,59003.3924343586,7238.578053236008,737.2700662612915,258.791455745697,675.399163722992]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, embb_trf2, 0.4, label = 'embb_trf2')
plt.bar(X_axis + 0.2, urllc_trf2, 0.4, label = 'urllc_trf2')
 
plt.xticks(X_axis, X,fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Configurations", fontsize=12)
plt.ylabel("Processing times (s)", fontsize=13)
plt.yscale("log")
plt.tight_layout()
# plt.show()
#^ NOTE: comment plt.show() and de-comment the next line to export a TiKZ version of the plot, as in the paper
#^       otherwise, the other way round to only see the plot
tikzplotlib.save("../results/motivation-results/processing-times/plot_rtx3090_proc-times-embb_urllc-trf2.tex")
plt.savefig("../results/motivation-results/processing-times/plot_rtx3090_proc-times-embb_urllc-trf2.png")
