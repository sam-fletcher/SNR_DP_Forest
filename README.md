# SNR_DP_Forest
A Differentially-Private Random Forest using Signal-to-Noise Ratio

Based on the algorithm proposed in: 

S. Fletcher and M. Z. Islam, A Differentially-Private Random Decision Forest using Reliable Signal-to-Noise Ratios. In *Proceedings of the 28th Australasian Joint Conference on Artificial Intelligence (AI 2015)*, Canberra, Australia, 30 Nov - 4 Dec, 2015, 
( *Lecture Notes in Computer Science (LNCS)*,  Vol. 9457, pp. 192-203, DOI: 10.1007/978-3-319-26350-2_17 )

The algorithm requires:

- training and testing categorical (i.e. discrete) data
- the total privacy budget

The algorithm outputs:

- a differentially-private classification model
- several variables that describe the model and its performance on the testing data

You can redistribute them and/or modify them under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. The main requirement of which is that you cite the above paper. Thanks!
