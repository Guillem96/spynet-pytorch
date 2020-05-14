from typing import NamedTuple, Tuple

MAX_G = 5


class BaseGConf(NamedTuple):
    image_size: Tuple[int, int] = (24, 32)


class GConf(object):

    def __init__(self, level: int) -> None:
        assert level >= 0 and level <= MAX_G
        self.base_conf = BaseGConf()
        self.scale = 2 ** level

    @property
    def image_size(self):
        return (self.base_conf.image_size[0] * self.scale, 
                self.base_conf.image_size[1] * self.scale)
