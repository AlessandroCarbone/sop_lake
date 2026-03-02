import logging, os
from datetime           import datetime
from data_io            import clean_folder
from dmft_config        import load_sim_config
from dmft_simulation    import dmft_simulation

clean_folder(fig_dir="figures")

def setup_logging():
    logging.basicConfig(
        filename="log.txt",                 # file name
        filemode="w",                       # writing mode
        level=logging.INFO,                 # minimum level: info, debug, warning
        format="%(message)s"
    )

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("DMFT")

    output_file_names = {
        "dmft": "dmft_output.json",
        "vemb": "vemb_output.json",
        "conv": "conv_output.json",
        "opt" : "opt_output.json"}
    
    config = load_sim_config("config.yaml")
    sim = dmft_simulation(config)
    start_time = datetime.now()
    logger.info("========================================")
    logger.info("Start DMFT simulation - %s",start_time)
    logger.info("========================================")
    sim.run(output_file_names=output_file_names)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info("========================================")
    logger.info("End DMFT simulation - %s",end_time)
    logger.info("Total runtime: %s",elapsed_time)
    logger.info("========================================")
