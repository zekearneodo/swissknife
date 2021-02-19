import os
import logging
import pandas as pd 


from swissknife.hilevel import metricskeras as mk 
from swissknife.hilevel import ffnnkeras as ffnn

logger = logging.getLogger('swissknife.hilevel.traintest')

def mot_wise_model(sess_data, mots_per_run):
    fit_type = 'model'
    mots_per_run = 10

    all_mots = sess_data.all_mots_list()
    n_mots = all_mots.size
    chunks = [all_mots[i:i+mots_per_run] for i in np.arange(0, n_mots, mots_per_run)]
    n_chunks = len(chunks)

    all_model_pred = []
    all_Z_t = []
    all_Y_t = []
    all_X_t = []

    all_syn = []
    all_neur = []
    all_bos = []

    train_pd_path = os.path.join(sess_data.fn['folders']['ss'], 
    'train_pd_{}_{}_{}.pkl'.format(sess_data.fp['network'], 
                                                                                sess_data.fp['unit_type'],
                                                                                fit_type)
                                                                                )
        
    for i_chunk, chunk in enumerate(chunks[:]):
        logger.info('Train/Test for chunk {}/{}'.format(i_chunk, n_chunks))
        X, Y, X_t, Y_t, Z, Z_t = sess_data.gimme_motif_set(chunk)
        print(X.shape)
        print(X_t.shape)
        
        mod_pred = ffnn.train_test_ffnn_model(X, Y, Z, X_t, Y_t, Z_t, sess_data)
        
        logger.info('Integrating the model to generate synthetic motifs')
        
        syn_fit, par_fit = mk.pred_par_to_song(Y_t, sess_data.fp)
        syn_pred, par_pred = mk.pred_par_to_song(mod_pred, sess_data.fp)
        bos = sess_data.gimme_raw_stream(Z_t)
        
        all_model_pred.append(mod_pred)
        
        all_Z_t.append(Z_t)
        all_Y_t.append(Y_t)
        all_X_t.append(X_t)
        all_syn.append(syn_fit)
        all_neur.append(syn_pred)
        all_bos.append(bos)
        
        plt.plot(((mod_pred[:200, :])));
        plt.plot(Y_t[:200], '--');
        

    training_pd = pd.DataFrame({
        'Y_p': all_model_pred,
        'Z_t': all_Z_t,
        'Y_t': all_Y_t,
        'X_t': all_X_t,
        'syn_t': all_syn,
        'neur_t': all_neur,
        'bos_t': all_bos
        })


    training_pd.to_pickle(train_pd_path)
    logger.info('saved all network chunks data/metadata in {}'.format(train_pd_path))