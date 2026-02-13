class AppState:
    def __init__(self):
        self.raw_df = None
        self.column_meta = None
        self.encoded_df = None
        self.filename = None
        self.taxonomy = None

    def reset(self):
        self.__init__()

    def is_loaded(self):
        return self.raw_df is not None


app_state = AppState()
