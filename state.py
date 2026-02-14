class AppState:
    def __init__(self):
        self.raw_df = None
        self.column_meta = None
        self.encoded_df = None
        self.filename = None
        self.taxonomy = None
        self.reset_wsom()

    def reset(self):
        self.__init__()

    def reset_wsom(self):
        self.wsom_training = False
        self.wsom_progress = 0
        self.wsom_total_epochs = 0
        self.wsom_error = None
        self.wsom_parent_id = None
        self.wsom_proposed_ids = []
        self.wsom_thread = None

    def is_loaded(self):
        return self.raw_df is not None


app_state = AppState()
