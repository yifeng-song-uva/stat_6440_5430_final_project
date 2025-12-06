from variational_inference_sLDA_E_step import *
import multiprocessing

def list_to_dict(l, start_indx, end_indx):
    return {i:l[i] for i in range(start_indx, end_indx)}

def e_step_wrapper(q):
    # Wrapper for the individual processors which is necessary for parallel processing using the multiprocessing library
    e_step = VI_sLDA_E_Step(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], epsilon=q[9])
    e_step.coordinate_ascent_training()
    e_step.save_parameters()

if __name__ == '__main__':
    m = 10 # Parallelization: each cpu core will process m documents
    # whenever this script is called in the Jupyter Notebook "train_batch_VI_sLDA_movie_rating.ipynb",
    # it accesses those variables defined in the name space of that Notebook.
    # Thus, within each iteration of the variational EM procedure for sLDA, the E step incorporates
    # the most up-to-date global parameters from the output of the M step from the previous iteration
    query_list = [(fpath, K, {j:input_data_x[j] for j in range(i,i+m)}, list_to_dict(input_data_y, i, i+m), new_alpha, new_eta, new_delta, new_Lambda, epsilon) for i in range(0,len(input_data_y)-m,m)]
    r = len(input_data_y) % m
    if r == 0:
        i = len(input_data_y) - m
    else:
        i = len(input_data_y) - r
    query_list.append((fpath, K, {j:input_data_x[j] for j in range(i,len(input_data_y))}, list_to_dict(input_data_y,i,len(input_data_y)), new_alpha, new_eta, new_delta, new_Lambda, epsilon))
    query_list = [(indx,) + q for indx,q in enumerate(query_list)] 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(e_step_wrapper, query_list)