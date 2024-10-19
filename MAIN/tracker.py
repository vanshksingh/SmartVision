# tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
import traceback

def initialize_tracker(config):
    """
    Initialize the Deep SORT tracker.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        DeepSort: Initialized DeepSort tracker instance.
    """
    try:
        tracker = DeepSort(max_age=30,
                           n_init=3,
                           nn_budget=100,
                           max_cosine_distance=config["TRACKER_MAX_DISTANCE"],
                           override_track_class=None)
        logging.info("Deep SORT tracker initialized successfully.")
        return tracker
    except Exception as e:
        logging.error(f"Error initializing tracker: {e}")
        logging.debug(traceback.format_exc())  # Log full traceback for debugging
        raise e  # Re-raise the exception after logging
