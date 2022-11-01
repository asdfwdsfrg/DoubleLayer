import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network() #new network from network_module
    return network
