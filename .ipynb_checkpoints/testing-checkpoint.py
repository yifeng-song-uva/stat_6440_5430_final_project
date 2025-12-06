import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing
import datetime as dt

class Term_GPA_and_Credits_Processor(object):
    # The purpose of this class object is to calculate the GPA/Enrollment_Intensity of each student x term based on the prior term GPA and term credits earned data prior to this term

    def __init__(self, indx, ol):
        self.indx = indx
        self.ol = ol # outer list, a list of tuples
        
    def find_slope(self):
        self.result = np.hstack([np.random.normal(loc=self.ol[j][0], scale=self.ol[j][1], size=10) for j in range(len(self.ol))])
            
    def save(self):
        # The output of parallel processing will be individual pickled dictionary files corresponding to each student x term: they'll be merged in later steps
        pickle.dump(self.result, open("values/{}.pickle".format(self.indx), "wb"))
        
def term_gpa_and_credits_processor_wrapper(q):
    # Wrapper for the individual processors which is necessary for parallel processing using the multiprocessing library
    tgcp = Term_GPA_and_Credits_Processor(q[0], q[1])
    tgcp.find_slope()
    tgcp.save()
    
    
if __name__ == '__main__':
    # First create GPA trend predictors
    # values = [(e,1) for e in np.random.normal(0,10,1000)]
    query_list = [values[i:(i+5)] for i in range(0,len(values), 5)] # Parallelization: each cpu core will process 5 documents
    query_list = [(indx, q) for indx,q in enumerate(query_list)] 
    pool = multiprocessing.Pool(multiprocessing.cpu_count()) # Use parallel processing to speed up this script, as the trendline predictors of each student x term can be constructed independently
    pool.map(term_gpa_and_credits_processor_wrapper, query_list)