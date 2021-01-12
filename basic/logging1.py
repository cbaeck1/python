import logging

logger = logging.getLogger("logging_sample")
logger.setLevel(logging.DEBUG)

# file log handler
fh = logging.FileHandler("logging_sample.log")
fh.setLevel(logging.ERROR)
# console log handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

#create formmater
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
logger.debug("this is debugging")
logger.info("this is info")

