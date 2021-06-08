import configparser

def load_config_file(config_file = "pygizmo.cfg"):
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#'))
    cfg.optionxform=str
    cfg.read(config_file)
    return cfg

cfg = load_config_file()
