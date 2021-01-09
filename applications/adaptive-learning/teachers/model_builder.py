import logging


def build(config):
    """ build a teacher model based on config file"""
    model_name = config.get_section_dict('Teacher')['Name']
    if model_name == "iterdet":
        from iterdet import IterDet
        model = IterDet(config=config)
        logging.info("model loaded successfully")
    elif model_name == "faster_rcnn_nas":
        from faster_rcnn_nas import FasterRcnnNas
        model = FasterRcnnNas(config=config)
        logging.info("model loaded successfully")
    else:
        raise ValueError("The Teacher model name should be iterdet but {} provided".format(model_name))
    return model
