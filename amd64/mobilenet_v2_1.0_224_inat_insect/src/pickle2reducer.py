from multiprocessing.reduction import ForkingPickler, AbstractReducer


class ForkingPickler2(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=2):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=2):
    ForkingPickler2(file, protocol).dump(obj)


class Pickle2Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler2
    register = ForkingPickler2.register
    dump = dump
