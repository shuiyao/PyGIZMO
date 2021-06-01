import configparser
cfg = configparser.ConfigParser(inline_comment_prefixes=('#'))
cfg.optionxform=str
cfg.read("pygizmo.cfg")
