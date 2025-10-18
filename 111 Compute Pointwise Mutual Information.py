import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
	a = joint_counts/total_samples

    b = (total_counts_x/total_samples)
    c = (total_counts_y/total_samples)
    d = b*c

    p = a/d

    pmi = np.log2(p)

    return round(pmi, 3)
