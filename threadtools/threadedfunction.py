import multiprocessing
import logging

logger = logging.getLogger('threadtools.threadedfunction')


class threadedFunction(multiprocessing.Process):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, join_threads=[], verbose=None):

        multiprocessing.Process.__init__(self, group=group,
                                    target=target,
                                    name=name)

        self.func = args[0]
        self.args = args[1:]
        self.kwargs = kwargs
        self.out = None
        self.join_threads = join_threads
        return

    def run(self):
        logger.info('Starting function {0} in thread {1}'.format(self.func.func_name,
                                                                 self.name))

        # for arg in self.args:
        #     print arg
        # for key, arg in self.kwargs.iteritems():
        #     print key
        #     print arg
        self.out = self.func(*self.args, **self.kwargs)
        logger.info("done process {0}".format(self.name))
        return self.out
