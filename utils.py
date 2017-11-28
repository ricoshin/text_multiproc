from time import time, strftime, gmtime

class StopWatch(object):
    time = dict()
    history = dict()

    def __init__(self, name):
        self.name = name
        StopWatch.go(name)

    def __enter__(self):
        pass

    def __exit__(self, type, value, trace_back):
        StopWatch.stop(self.name)
        del self.name

    @classmethod
    def go(cls, name):
        cls.time[name] = time()

    @classmethod
    def stop(cls, name):
        start_time = cls.time.get(name, None)
        if start_time:
            elapsed_time = time() - start_time
            cls.print_elapsed_time(name, elapsed_time)
            cls.history[name] = elapsed_time
            del cls.time[name]
        else:
            print('Not registered name : ', name)

    @classmethod
    def print_elapsed_time(cls, name, seconds):
        msg = "StopWatch [%s] : %5f" % (name, seconds)
        hms = strftime("(%Hhrs %Mmins %Ssecs)", gmtime(seconds))
        print(msg, hms)


class Config(object):
    def __init__(self, cfg=None):
        if cfg is not None:
            self.update(cfg)

    def update(self, new_config):
        self.__dict__.update(new_config)

    def __repr__(self):
        return self.__dict__.__repr__()
