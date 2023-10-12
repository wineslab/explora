"""
Plot SHAP computing times

10/13/2023
_______________
    
Summary: plotting for the "urllc-trf2" configuration the computing times 
         on RTX 3090 and A100.
         This is the plot in figure 4(a) of the manuscript.
      
DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib 

X = ['37','38','39','40','41','42'] # this is the urllc-trf2 configuration
#^ computing times taken from /results/shap_explanations/processing_times/gpu-rtx_3090
urartu = [60043.459069252014,59003.3924343586,7238.578053236008,737.2700662612915,258.791455745697,675.399163722992]
#^ computing times taken from /results/shap_explanations/processing_times/gpu-a100
oracle = [88204.01963710785,87014.2811319828,10437.84526181221,1051.6206815242767,366.91124391555786,1024.8842041492462]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis, urartu, 0.8)
plt.bar(X_axis - 0.2, urartu, 0.4, label = 'RTX-3090')
plt.bar(X_axis + 0.2, oracle, 0.4, label = 'A-100')
 
plt.xticks(X_axis, X,fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Configurations", fontsize=12)
plt.ylabel("Processing times (s)", fontsize=12)
plt.yscale("log")
plt.tight_layout()
plt.legend()
# plt.show()
#^ NOTE: comment plt.show() and de-comment the next line to export a TiKZ version of the plot, as in the paper
#^       otherwise, the other way round to only see the plot
tikzplotlib.save("../results/motivation-results/processing-times/plot_a100_vs_rtx3090_proc-times-37_42.tex")
plt.savefig("../results/motivation-results/processing-times/plot_a100_vs_rtx3090_proc-times-37_42.png")
