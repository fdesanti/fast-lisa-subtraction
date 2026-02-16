import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, 
                    format='[%(levelname)s] - %(message)s')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

