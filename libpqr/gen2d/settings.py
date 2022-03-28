import typing
import numpy as np
from dataclasses import dataclass, fields


@dataclass
class VlbSettings:
    pretrain: bool=False
    training: bool=True
    rebuild_order: str="breadth_and_depth"
    def items(self):
        for k in fields(self):
            yield k.name, getattr(self, k.name)
    def __str__(self):
        s = ""
        for k, v in self.items():
            s += "Settings: %-20s = %s\n" % (k, str(v))
        return s[:-1]
            

def get_default_settings():
    return VlbSettings()

