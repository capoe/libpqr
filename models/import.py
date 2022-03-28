import torch

import libpqr.gen3d as gen3d
import libpqr.gen2d as gen2d


def import_state_dict(m, pt_file):
    pars = [(name,p) for (name, p) in m.state_dict().items() ]
    pars_ref = torch.load(pt_file)
    state = m.state_dict()
    print("Open", pt_file)
    for name, param in pars_ref.items():
        print("  [copy]", name)
        state[name].copy_(param)
    return m


def get_configure_settings():
    settings = gen3d.get_default_settings()
    settings.data_shuffle = True
    settings.invert_skiplist = False
    settings.crop_3d = True
    settings.perturb_pos = True
    settings.perturb_steps = 5
    settings.perturb_ampl = 0.5
    settings.cut_hyper_width = 0.5
    settings.cut_hyper_intra = 2.25
    settings.cut_hyper_inter = 7.5
    settings.cut_hyper_edge = 7.5
    settings.radial_weight_pivot = 6.5
    settings.radial_weight_decay = 0.1
    settings.radial_weight_const = 1.0
    settings.depth_2d = 4
    settings.baseline_type = "2d"
    return settings

if __name__ == "__main__":

    feat, m = gen3d.build_model(get_configure_settings(), torch.device("cpu"))
    m = import_state_dict(m, "model_r.arch.pt")
    for k, v in m.settings.items():
        print("Settings: %-20s = %s" % (k, str(v)))
    torch.save(m, "r/model_r.arch")

    feat, m = gen2d.build_model(torch.device("cpu"))
    m = import_state_dict(m, "model_q_pdbl.arch.pt")
    torch.save(m, "q/model_q_pdbl.arch")

    feat, m = gen2d.build_model(torch.device("cpu"))
    m = import_state_dict(m, "model_q_erd3.arch.pt")
    torch.save(m, "q/model_q_erd3.arch")
