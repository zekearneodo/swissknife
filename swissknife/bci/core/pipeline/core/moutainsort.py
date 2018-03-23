import json


from swissknife.bci.core import expstruct as et
from swissknife.bci.core.file import h5_functions as h5f

logger = logging.getLogger('swissknife.bci.core.pipeline.core.kilosort')


def make_moutainsort_config(bird, sess,
                            detect_sign=-1,
                            adjacency_radius=-1,
                            ):

    fn = et.file_names(bird, sess)
    exp_par = et.get_parameters(bird, sess)  # load the yml parameter file
    sort_dir = fn['folders']['ss']
    logger.debug('local sort dir: {}'.format(local_sort_dir))
    s_f = h5f.get_record_sampling_frequency(et.open_kwd(bird, sess))

    params = {
        'detect_sign': detect_sign,
        'samplerate': s_f,
    }
    logger.debug(params)

    et.mkdir_p(sort_dir)

    with open(os.path.join(dest, 'params.json'), 'w') as output:
        json.dump(output_args, output)
