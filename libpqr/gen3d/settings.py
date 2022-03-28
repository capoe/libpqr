import typing
from dataclasses import dataclass, field, fields


@dataclass
class LexSettings:
    data_skiplist: str="./data/complexes_skiplist.json"
    data_complexes: str="./data/complexes_paths_pdb.json"
    data_shuffle: bool=True
    invert_skiplist: bool=False
    crop_3d: bool=True
    perturb_pos: bool=True
    perturb_steps: int=5
    perturb_ampl: float=0.5
    cut_hyper_width: float=0.5   # 0.5
    cut_hyper_intra: float=2.25  # 2.25
    cut_hyper_inter: float=7.5   # 7.5
    cut_hyper_edge: float=7.5    # 7.5
    radial_weight_pivot: float=6.5
    radial_weight_decay: float=0.1
    radial_weight_const: float=1.0
    stage_final_only: bool=False
    stage_chains_only: bool=False
    stage_contacts_only: bool=False
    stage_contacts_cutoff: float=4.0
    stage_contacts_invert: bool=False
    stage_motif_size_cutoff: int=-1
    include_environment: bool=True
    depth_2d: int=4
    baseline_type: str="2d"

    def items(self):
        for k in fields(self):
            yield k.name, getattr(self, k.name)

    def __str__(self):
        s = ""
        for k, v in self.items():
            s += "Settings: %-20s = %s\n" % (k, str(v))
        return s[:-1]

    def update(self, other):
        for key, val in other.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise KeyError("No such field '%s' in settings" % key)
            

def get_default_settings():
    settings = LexSettings()
    return settings


